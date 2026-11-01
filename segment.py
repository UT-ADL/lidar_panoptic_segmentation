#!/usr/bin/env python3
"""Lidar panoptic segmentation of one scan with the ONNX model: per point ground / static / dynamic, object instances,
nuScenes classes and the over-ground velocity of dynamic objects (from the 4 previous scans).

usage: python segment.py example.pcd [--lidar_height 2.11] [--no-sweeps] [--show] [--plot out.png] [--out results.npz]
                         [--cpu]

The scan is a PCD file (ascii or binary) with the fields x y z intensity in the lidar frame (x forward, y left, z up,
origin at the lidar) and the sensor's raw intensity. Previous scans are read from <scan>_sweep_1.pcd .. _sweep_4.pcd
(1 = newest), already transformed into the current lidar frame, with their time lags (s) in sweeps.json
{"lag": [...]} next to the scan. --show opens an interactive Open3D view, --plot renders the same scene offscreen to a
PNG (both need `pip install open3d`; the printed object list is the legend). --cpu: CPU even if CUDA is available.
"""
import argparse, json, os
import numpy as np
import onnxruntime as ort

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pseg_v6.1_fp16.onnx')
MIN_RANGE = 2.7          # m, minimum range of the lidar driver the model was deployed with
MAX_HEIGHT = 2.5         # m above the local ground: higher non-ground points (canopy) are not obstacles
FILTER_RANGE = 38.0      # m, isolated static points (noise) are dropped within this range
PADDING = [1e4, 0.0, 0.0, 0.0, 0.0]   # padding row of the fixed-size inputs: a point far outside the model's grid
MOVING = 1.0             # m/s, objects faster than this get a velocity arrow
GROUPS = ['ground', 'static', 'dynamic']
PALETTE = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.58, 0.4, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76],
                    [0.74, 0.74, 0.13], [0.09, 0.75, 0.81], [0.68, 0.78, 0.91], [1.0, 0.73, 0.47], [0.6, 0.6, 0.0]])


def read_pcd(path):
    """x, y, z, intensity (N, 4) float32 of a PCD file (DATA ascii or binary; intensity 0 if the file has none)."""
    header = {}
    with open(path, 'rb') as f:
        while 'DATA' not in header:
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: no DATA line in the PCD header')
            parts = line.decode('ascii', 'replace').split()
            if parts and not parts[0].startswith('#'):
                header[parts[0].upper()] = parts[1:]
        fields = header['FIELDS']
        sizes = [int(s) for s in header['SIZE']]
        types = header['TYPE']
        counts = [int(c) for c in header.get('COUNT', ['1'] * len(fields))]
        n = int(header.get('POINTS', [int(header['WIDTH'][0]) * int(header.get('HEIGHT', ['1'])[0])])[0])
        kind = header['DATA'][0].lower()
        names = [f'{name}_{i}' if fields.count(name) > 1 else name for i, name in enumerate(fields)]
        if kind == 'binary':
            dtype = np.dtype([(name, f'<{t.lower()}{s}', (c,)) for name, s, t, c in zip(names, sizes, types, counts)])
            rec = np.frombuffer(f.read(n * dtype.itemsize), dtype, count=n)
            column = lambda name: rec[name][:, 0].astype(np.float32)   # noqa: E731
        elif kind == 'ascii':
            table = np.loadtxt(f, dtype=np.float64, ndmin=2)[:n]
            start = dict(zip(names, np.cumsum([0] + counts)[:-1]))
            column = lambda name: table[:, start[name]].astype(np.float32)   # noqa: E731
        else:
            raise ValueError(f'{path}: DATA {kind} is not supported (use ascii or binary)')
    zero = np.zeros(n, np.float32)
    return np.stack([column(k) if k in names else zero for k in ('x', 'y', 'z', 'intensity')], 1)


def padded(rows, size, width):
    out = np.tile(np.array(PADDING[:width], np.float32), (size, 1))
    if len(rows) > size:
        print(f'warning: {len(rows)} points, the model input holds {size}; the rest are left out')
    out[:min(len(rows), size)] = rows[:size]
    return out


def read_sweeps(scan_path, n_sweeps):
    """Previous scans next to the scan, with their time lag as a fifth column (newest first)."""
    base = os.path.splitext(scan_path)[0]
    lags = json.load(open(os.path.join(os.path.dirname(os.path.abspath(scan_path)), 'sweeps.json')))['lag']
    rows = []
    for k, lag in enumerate(lags[:n_sweeps], 1):
        s = read_pcd(f'{base}_sweep_{k}.pcd')
        rows.append(np.column_stack((s, np.full(len(s), lag, np.float32))))
    return np.concatenate(rows) if rows else np.zeros((0, 5), np.float32)


def session(cpu):
    options = ort.SessionOptions()
    if hasattr(os, 'sched_getaffinity'):   # the CPUs this process may use (containers, batch systems)
        options.intra_op_num_threads = len(os.sched_getaffinity(0))
    providers = ['CPUExecutionProvider'] if cpu else \
        [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in ort.get_available_providers()]
    return ort.InferenceSession(MODEL, options, providers=providers)


def instances(points, group, inst, native, velocity, names):
    """(id, class, number of points, centroid xyz, mean xy velocity) of the dynamic object instances."""
    out = []
    for i in np.unique(inst[(group == 2) & (inst > 0)]):
        m = (inst == i) & (group == 2)
        out.append((int(i), names[np.bincount(native[m]).argmax()], int(m.sum()), points[m, :3].mean(0), velocity[m].mean(0)))
    return out


def colours(group, inst):
    c = np.tile([0.6, 0.6, 0.6], (len(group), 1))              # ground
    c[group == 1] = [0.1, 0.45, 0.1]                            # static
    dyn = group == 2
    c[dyn] = np.where(inst[dyn, None] > 0, PALETTE[inst[dyn] % len(PALETTE)], [0.9, 0.1, 0.1])   # no instance: red
    return c


def scene(points, group, inst, objects):
    """Open3D geometries: the labelled points (group >= 0) and velocity arrows (1 s ahead) of moving objects."""
    import open3d as o3d
    keep = group >= 0
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[keep, :3].astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(colours(group[keep], inst[keep]))
    ends, lines = [], []
    for _, _, _, c, v in objects:
        if np.hypot(*v) <= MOVING:
            continue
        tip = c + [v[0], v[1], 0.0]
        side = np.array([-v[1], v[0], 0.0]) / np.hypot(*v) * 0.4
        back = c + 0.8 * np.array([v[0], v[1], 0.0])
        k = len(ends)
        ends += [c, tip, back + side, back - side]
        lines += [[k, k + 1], [k + 1, k + 2], [k + 1, k + 3]]
    arrows = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(ends, np.float64).reshape(-1, 3)),
                                  o3d.utility.Vector2iVector(np.array(lines, np.int32).reshape(-1, 2)))
    arrows.paint_uniform_color([0.0, 0.0, 0.0])
    return cloud, arrows


def camera(lidar_height):
    """Ego-centred view from slightly above and behind the car: (centre, eye, up)."""
    return [14.0, 0.0, -lidar_height], [-14.0, 0.0, 11.0], [0.0, 0.0, 1.0]


def show(points, group, inst, objects, lidar_height):
    import open3d as o3d
    cloud, arrows = scene(points, group, inst, objects)
    vis = o3d.visualization.Visualizer()
    vis.create_window('lidar panoptic segmentation (object list: see the printed output)', 1400, 900)
    vis.add_geometry(cloud)
    vis.add_geometry(arrows)
    vis.get_render_option().point_size = 2.0
    centre, eye, up = camera(lidar_height)
    view = vis.get_view_control()
    view.set_lookat(centre)
    view.set_front(np.subtract(eye, centre) / np.linalg.norm(np.subtract(eye, centre)))
    view.set_up(up)
    view.set_zoom(0.3)
    vis.run()
    vis.destroy_window()


def plot(points, group, inst, objects, lidar_height, path):
    import open3d as o3d
    cloud, arrows = scene(points, group, inst, objects)
    renderer = o3d.visualization.rendering.OffscreenRenderer(1600, 1000)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.view.set_post_processing(False)   # exact colours (no tone mapping)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader, mat.point_size = 'defaultUnlit', 3.0
    line = o3d.visualization.rendering.MaterialRecord()
    line.shader, line.line_width = 'unlitLine', 3.0
    renderer.scene.add_geometry('points', cloud, mat)
    renderer.scene.add_geometry('velocity', arrows, line)
    centre, eye, up = camera(lidar_height)
    renderer.setup_camera(60.0, centre, eye, up)
    o3d.io.write_image(path, renderer.render_to_image())
    print('saved', path, '(the object list above is the legend: ids, classes)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scan')
    ap.add_argument('--lidar_height', type=float, default=2.11, help='lidar height above the ground (m)')
    ap.add_argument('--no-sweeps', action='store_true', help='without previous scans (no usable velocity)')
    ap.add_argument('--show', action='store_true', help='interactive Open3D view')
    ap.add_argument('--plot', default=None, help='render the scene to this PNG (Open3D offscreen)')
    ap.add_argument('--out', default=None, help='save the per-point results (npz)')
    ap.add_argument('--cpu', action='store_true', help='CPU even if CUDA is available')
    a = ap.parse_args()

    sess = session(a.cpu)
    meta = sess.get_modelmeta().custom_metadata_map
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    names = json.loads(meta['class_names'])
    scan = read_pcd(a.scan)
    scan = scan[np.isfinite(scan).all(1) & (np.hypot(scan[:, 0], scan[:, 1]) >= MIN_RANGE)]
    inputs = dict(points=padded(scan, shapes['points'][0], 4), max_height=np.array([MAX_HEIGHT], np.float32),
                  lidar_height=np.array([a.lidar_height], np.float32), filter_range=np.array([FILTER_RANGE], np.float32))
    n_sweeps = json.loads(meta.get('sweeps', '0'))
    if 'sweeps' in shapes:
        sweeps = np.zeros((0, 5), np.float32) if a.no_sweeps else read_sweeps(a.scan, n_sweeps)
        inputs['sweeps'] = padded(sweeps, shapes['sweeps'][0], 5)
    outputs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, inputs)))

    n = min(len(scan), shapes['points'][0])
    points, group, inst, native = scan[:n], outputs['group'][:n], outputs['inst'][:n], outputs['native'][:n]
    velocity = outputs['velocity'][:n]
    print(f'{os.path.basename(MODEL)} on {sess.get_providers()[0]}: {n} points, '
          f'{0 if a.no_sweeps or "sweeps" not in shapes else n_sweeps} previous scans')
    print('  ' + ', '.join(f'{name} {np.sum(group == g)}' for g, name in enumerate(GROUPS)) +
          f', none {np.sum(group < 0)} (outside the grid, above max_height or isolated)')
    objects = instances(points, group, inst, native, velocity, names)
    print(f'{len(objects)} objects (dynamic points without an instance: {np.sum((group == 2) & (inst == 0))}):')
    for i, cls, count, _, v in objects:
        print(f'  {i:3d}  {cls:36s} {count:5d} points  speed {np.hypot(*v):4.1f} m/s')
    if a.out:
        np.savez_compressed(a.out, points=points, group=group, inst=inst, native=native, velocity=velocity,
                            class_names=np.array(names))
        print('saved', a.out)
    if a.plot:
        plot(points, group, inst, objects, a.lidar_height, a.plot)
    if a.show:
        show(points, group, inst, objects, a.lidar_height)


if __name__ == '__main__':
    main()
