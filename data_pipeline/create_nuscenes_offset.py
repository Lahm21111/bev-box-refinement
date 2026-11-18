"""Create a random nuScenes box offset and directly usable refinement inputs."""
import argparse
from pathlib import Path
import secrets

import numpy as np

from model_runtime.gemini_box_decision import create_run_directory, read_json, write_json
from model_tools.actions import build_actions
from model_tools.geometry import perturb_box
from model_tools.nuscenes_views import sample_identity, project_corners
from model_tools.render_views import render_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/nuscenes_box_crops'))
    parser.add_argument('--output', type=Path, default=Path('outputs/nuscenes_random_offset'))
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    meta, mapping = read_json(args.source / 'metadata.json'), read_json(args.source / 'pixel_mapping.json')
    if meta.get('dataset') != 'nuScenes':
        parser.error('--source must be a nuScenes visualization export')
    ppm = mapping['views']['lidar_top']['pixels_per_meter']
    with np.load(args.source / 'geometry.npz') as geometry:
        original = geometry['corners_lidar'].copy()
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(seed)
    bounds = np.array(mapping['bounds_lidar'])
    rect = np.array(mapping['views']['camera_crop']['crop_xyxy'])
    for _ in range(100):
        x, y = map(int, rng.choice([-30, -20, 20, 30], 2))
        yaw = int(rng.choice([-5, -2, 2, 5]))
        moved = perturb_box(original, x / ppm, -y / ppm, yaw)
        pixels = project_corners(moved, meta)
        if (np.all(moved >= bounds[0]) and np.all(moved <= bounds[1])
                and np.all(pixels >= rect[:2]) and np.all(pixels < rect[2:])):
            break
    else:
        raise ValueError('Could not find an offset inside the fixed four-view crop')
    output = create_run_directory(args.output)
    state = dict(**sample_identity(meta), seed=seed, pixels_per_meter=ppm,
        offset=dict(dx_m=x / ppm, dy_m=-y / ppm, dyaw_deg=yaw),
        original_corners_lidar=original.tolist(), perturbed_corners_lidar=moved.tolist(),
        offset_bev_pixels=dict(dx_px=x, dy_px=y, dyaw_deg=yaw))
    write_json(output / 'offset.json', state)
    reference_state = dict(state, offset=dict(dx_m=0.,dy_m=0.,dyaw_deg=0.),
                           perturbed_corners_lidar=original.tolist(),
                           offset_bev_pixels=dict(dx_px=0,dy_px=0,dyaw_deg=0))
    write_json(output / 'reference_state.json', reference_state)
    actions = build_actions(ppm)
    write_json(output / 'actions.json', dict(count=len(actions), pixels_per_meter=ppm, actions=actions))
    render_views(args.source, output / 'offset.json', None, output / 'views',
                 reference_mapping=args.source / 'pixel_mapping.json')
    render_views(args.source, output / 'offset.json', None, output / 'comparison',
                 reference_mapping=args.source / 'pixel_mapping.json', previous_offset=output / 'reference_state.json')
    (output / 'README.md').write_text(
        '# Random nuScenes offset\n\n'
        f'Seed: {seed}. BEV dx={x}px, dy={y}px, yaw={yaw}deg.\n\n'
        '`views/` contains clean current-box images for the model. `comparison/` shows\n'
        'the correct reference box in yellow dashes and the shifted box in red.\n'
        'The original source export is the correct same-target reference.\n'
        'Numeric perturbations stay in local metadata and are not sent to the model.\n')
    print(f'Seed {seed}: BEV dx={x:+d}px, dy={y:+d}px, yaw={yaw:+d}deg')
    print(f'Saved to {output.resolve()}')


if __name__ == '__main__':
    main()
