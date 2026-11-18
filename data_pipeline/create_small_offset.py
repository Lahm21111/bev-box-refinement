"""Create a small perturbation and 125 actions including a delete decision."""
import argparse
import json
import secrets
from pathlib import Path

import numpy as np
from model_tools.geometry import perturb_box
from model_tools.actions import build_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('outputs/offset_clean_views/pixel_mapping.json'))
    parser.add_argument('--source', type=Path, default=Path('outputs/box_crops'))
    parser.add_argument('--output', type=Path, default=Path('outputs/bev_small_offset'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dx-px', type=float, help='Explicit initial BEV X offset, positive right')
    parser.add_argument('--dy-px', type=float, help='Explicit initial BEV Y offset, positive down')
    parser.add_argument('--yaw-deg', type=float, help='Explicit initial yaw offset, positive counterclockwise')
    args = parser.parse_args()
    explicit = [args.dx_px, args.dy_px, args.yaw_deg]
    if any(value is not None for value in explicit):
        if any(value is None for value in explicit) or not np.isfinite(explicit).all():
            parser.error('Supply finite --dx-px, --dy-px, and --yaw-deg together')
    reference = json.loads(args.reference.read_text())
    meta = json.loads((args.source / 'metadata.json').read_text())
    if (reference['frame'], reference['box_id']) != (meta['frame'], meta['box_id']):
        raise ValueError('Reference belongs to another sample')
    ppm = reference['views']['lidar_top']['pixels_per_meter']
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(seed)
    px, py = map(int, rng.choice([-10, -5, 5, 10], 2))
    yaw = int(rng.choice([-1, 1]))
    if args.dx_px is not None:
        px, py, yaw = explicit
    # Pixel Y points down; LiDAR Y points up in the BEV image.
    dx, dy = px / ppm, -py / ppm
    with np.load(args.source / 'geometry.npz') as geometry:
        original = geometry['corners_lidar'].copy()
    shifted = perturb_box(original, dx, dy, yaw)
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(route=meta['route'], frame=meta['frame'], box_id=meta['box_id'],
                  seed=seed, pixels_per_meter=ppm,
                  offset=dict(dx_m=dx, dy_m=dy, dyaw_deg=yaw),
                  offset_bev_pixels=dict(dx_px=px, dy_px=py, dyaw_deg=yaw),
                  correction_bev_pixels=dict(dx_px=-px, dy_px=-py, dyaw_deg=-yaw),
                  pixel_convention='BEV: X right, Y down; yaw positive rotates LiDAR +X toward +Y',
                  original_corners_lidar=original.tolist(),
                  perturbed_corners_lidar=shifted.tolist())
    (args.output / 'offset.json').write_text(json.dumps(result, indent=2))
    actions = build_actions(ppm)
    (args.output / 'actions.json').write_text(json.dumps(dict(
        count=len(actions), reference_mapping=str(args.reference), pixels_per_meter=ppm,
        pixel_convention=result['pixel_convention'],
        execution=('adjust: rotate about CURRENT box center, then translate in LiDAR XY; '
                   'delete: remove current box and stop'),
        actions=actions), indent=2))
    print(json.dumps({k: result[k] for k in ['seed', 'offset', 'offset_bev_pixels',
                                            'correction_bev_pixels']}, indent=2))


if __name__ == '__main__':
    main()
