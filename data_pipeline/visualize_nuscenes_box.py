"""Export a nearby annotated nuScenes car as clean camera and LiDAR crops."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from model_runtime.gemini_box_decision import create_run_directory, write_json
from model_tools.geometry import transform
from model_tools.render_views import render_lidar, draw_box

DEFAULT_ROOT = Path('/inspire/hdd/project/feelingai/chenwenming-25012/jxs/zliu/data/nuscenes_mini')


def pose_matrix(record):
    """nuScenes quaternion order is [w,x,y,z]; return local-to-parent SE(3)."""
    w, x, y, z = np.asarray(record['rotation'], dtype=float) / np.linalg.norm(record['rotation'])
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = record['translation']
    return matrix


def box_corners(annotation):
    width, length, height = annotation['size']
    local = np.array([[x*length/2, y*width/2, z*height/2]
                      for x in [1, -1] for y in [1, -1] for z in [1, -1]])
    return transform(local, pose_matrix(annotation))


def camera_project(world, world_to_camera, intrinsic):
    # nuScenes cameras already use optical XYZ: right, down, forward.
    camera = transform(world, world_to_camera)
    homogeneous = camera @ np.asarray(intrinsic).T
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, camera[:, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--sample-token')
    parser.add_argument('--box-token')
    parser.add_argument('--allow-night', action='store_true', help='Include night scenes in automatic selection')
    parser.add_argument('--output', type=Path, default=Path('outputs/nuscenes_box_crops'))
    parser.add_argument('--padding', type=float, default=.2)
    parser.add_argument('--max-size', type=int, default=1024)
    args = parser.parse_args()
    if args.padding <= 0 or args.max_size < 64:
        parser.error('--padding must be positive and --max-size at least 64')
    tables = {}
    for name in ['sample', 'sample_data', 'sample_annotation', 'instance', 'category',
                 'sensor', 'calibrated_sensor', 'ego_pose', 'scene']:
        path = args.data_root / args.version / f'{name}.json'
        tables[name] = {item['token']: item for item in json.loads(path.read_text())}
    sensor_data = {}
    for record in tables['sample_data'].values():
        if record['is_key_frame']:
            calibration = tables['calibrated_sensor'][record['calibrated_sensor_token']]
            channel = tables['sensor'][calibration['sensor_token']]['channel']
            sensor_data.setdefault(record['sample_token'], {})[channel] = record

    def sensor_pose(record):
        calibration = tables['calibrated_sensor'][record['calibrated_sensor_token']]
        ego = tables['ego_pose'][record['ego_pose_token']]
        return pose_matrix(ego) @ pose_matrix(calibration)

    candidates = []
    for annotation in tables['sample_annotation'].values():
        if args.box_token and annotation['token'] != args.box_token:
            continue
        if args.sample_token and annotation['sample_token'] != args.sample_token:
            continue
        scene = tables['scene'][tables['sample'][annotation['sample_token']]['scene_token']]
        if not args.box_token and not args.sample_token and not args.allow_night and 'night' in scene['description'].lower():
            continue
        category = tables['category'][tables['instance'][annotation['instance_token']]['category_token']]['name']
        if category != 'vehicle.car':
            continue
        sensors = sensor_data.get(annotation['sample_token'], {})
        if not all(key in sensors for key in ['CAM_FRONT', 'LIDAR_TOP']):
            continue
        camera, lidar = sensors['CAM_FRONT'], sensors['LIDAR_TOP']
        intrinsic = tables['calibrated_sensor'][camera['calibrated_sensor_token']]['camera_intrinsic']
        world = box_corners(annotation)
        pixels, depth = camera_project(world, np.linalg.inv(sensor_pose(camera)), intrinsic)
        corners = transform(world, np.linalg.inv(sensor_pose(lidar)))
        distance = float(np.linalg.norm(corners.mean(0)[:2]))
        size = np.array([camera['width'], camera['height']])
        if (np.min(depth) <= .1 or not np.isfinite(pixels).all()
                or np.any(pixels < 10) or np.any(pixels > size-10)):
            continue
        if not args.box_token and (not 4 <= distance <= 18 or annotation['num_lidar_pts'] < 50
                                   or int(annotation['visibility_token']) < 3):
            continue
        if not all((args.data_root / record['filename']).exists() for record in [camera, lidar]):
            continue
        candidates.append((distance, -annotation['num_lidar_pts'], annotation, camera, lidar,
                           world, corners, pixels, intrinsic))
    if not candidates:
        raise ValueError('No nearby annotated car fully visible in CAM_FRONT; check data/version or select another box')
    _, _, annotation, camera, lidar, world, corners, pixels, intrinsic = min(candidates, key=lambda c: c[:2])
    points = np.fromfile(args.data_root / lidar['filename'], dtype=np.float32).reshape(-1, 5)[:, :3]
    low, high = corners.min(0), corners.max(0)
    margin = np.maximum((high-low)*args.padding, .15)
    bounds = np.stack([low-margin, high+margin])
    crop = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
    if not len(crop):
        raise ValueError('No LiDAR returns in the selected box neighborhood')
    output = create_run_directory(args.output)
    views = {}
    # Native nuScenes LiDAR X is right, Y is forward, Z is up.
    # Front is XZ and side is YZ, unlike the CARLA LiDAR axis order.
    for name, axes in [('lidar_top', [0, 1]), ('lidar_front', [0, 2]), ('lidar_side', [1, 2])]:
        image, views[name] = render_lidar(crop, corners, bounds, axes, args.max_size)
        image.save(output / f'{name}.png')
    with Image.open(args.data_root / camera['filename']) as original:
        image = original.convert('RGB')
    pad = np.ptp(pixels, axis=0)*args.padding
    rect = np.r_[np.floor(np.maximum(pixels.min(0)-pad, 0)),
                 np.ceil(np.minimum(pixels.max(0)+pad, image.size))].astype(int)
    draw_box(ImageDraw.Draw(image), pixels, '#ff604b', 3)
    image.save(output / 'camera_full.png')
    image.crop(tuple(rect)).save(output / 'camera_crop.png')
    views['camera_crop'] = dict(size_wh=(rect[2:]-rect[:2]).tolist(), crop_xyxy=rect.tolist(),
                                corners_pixels=(pixels-rect[:2]).tolist())
    sample = tables['sample'][annotation['sample_token']]
    metadata = dict(dataset='nuScenes', version=args.version, data_root=str(args.data_root.resolve()),
        scene=tables['scene'][sample['scene_token']]['name'], sample_token=sample['token'],
        box_token=annotation['token'], category='vehicle.car', box_source='dataset ground truth annotation',
        distance_lidar_xy_m=float(np.linalg.norm(corners.mean(0)[:2])),
        annotation_num_lidar_points=annotation['num_lidar_pts'], crop_points=len(crop),
        lidar_timestamp=lidar['timestamp'], camera_timestamp=camera['timestamp'],
        timestamp_difference_ms=(camera['timestamp']-lidar['timestamp'])/1000,
        lidar_filename=lidar['filename'], camera_filename=camera['filename'],
        coordinates='Native nuScenes LiDAR: X right, Y forward, Z up; camera optical: X right, Y down, Z forward',
        annotation=annotation, lidar_to_global=sensor_pose(lidar).tolist(),
        global_to_camera=np.linalg.inv(sensor_pose(camera)).tolist(), camera_intrinsic=intrinsic)
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'pixel_mapping.json', dict(dataset='nuScenes', frame=sample['token'],
        box_id=annotation['token'], bounds_lidar=bounds.tolist(), views=views,
        coordinates=metadata['coordinates'], image_content='only current red box and point cloud / camera RGB'))
    np.savez_compressed(output / 'geometry.npz', points=crop, corners_lidar=corners,
        corners_world=world, corners_camera_pixels=pixels, crop_bounds_lidar=bounds)
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>nuScenes car crops</title>'
        '<style>body{background:#17202b;color:white;font-family:sans-serif}'
        'main{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{max-width:100%;max-height:70vh}</style>'
        '<main>' + ''.join(f'<section><p>{label}</p><a href="{name}.png"><img src="{name}.png"></a></section>'
            for name, label in [('camera_crop','Front camera'),('lidar_top','BEV XY'),
                                ('lidar_front','Front XZ'),('lidar_side','Side YZ')]) + '</main>')
    (output / 'README.md').write_text(
        '# nuScenes car crops\n\nOriginal annotated car box, no perturbation.\n\n'
        'Native LiDAR axes differ from Bench2Drive: X right, Y forward, Z up.\n'
        'BEV = XY; front = XZ; side = YZ. All PNGs have no axes, titles or legends.\n'
        'Camera and LiDAR each use their own ego pose and calibration.\n'
        'Camera-LiDAR timestamp difference is recorded in metadata.json; moving objects\n'
        'can retain small sensor-time differences. This export can be used as source/reference for the refinement pipeline.\n'
)
    print(json.dumps({key: metadata[key] for key in ['scene', 'sample_token', 'box_token',
        'distance_lidar_xy_m', 'annotation_num_lidar_points', 'crop_points', 'timestamp_difference_ms']}, indent=2))
    print(f'Saved to {output.resolve()}')


if __name__ == '__main__':
    main()
