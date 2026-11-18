"""Apply a reproducible random XY/yaw perturbation to the current sample box."""
import argparse
import json
import secrets
from pathlib import Path

from model_tools.geometry import EDGES, transform, perturb_box
import matplotlib
matplotlib.use('Agg')
import laspy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/box_crops'))
    parser.add_argument('--data-root', type=Path, default=Path('data/bench2drive'))
    parser.add_argument('--output', type=Path, default=Path('outputs/bev_random_offset'))
    parser.add_argument('--max-xy', type=float, default=1.0,
                        help='Uniform translation range [-max-xy, max-xy] per axis, meters')
    parser.add_argument('--max-yaw-deg', type=float, default=15.0)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if (not np.isfinite([args.max_xy, args.max_yaw_deg]).all()
            or min(args.max_xy, args.max_yaw_deg) < 0):
        parser.error('Perturbation limits must be finite and nonnegative')
    meta = json.loads((args.source / 'metadata.json').read_text())
    with np.load(args.source / 'geometry.npz') as geometry:
        original = geometry['corners_lidar'].copy()
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(seed)
    dx, dy = rng.uniform(-args.max_xy, args.max_xy, 2)
    yaw = rng.uniform(-args.max_yaw_deg, args.max_yaw_deg)
    shifted = perturb_box(original, dx, dy, yaw)
    both = np.concatenate([original, shifted])
    low, high = both.min(0), both.max(0)
    margin = np.maximum((high - low) * .2, .3)
    bounds = np.stack([low - margin, high + margin])
    # Reload the scan so the expanded crop includes real surrounding points.
    path = args.data_root / meta['route'] / 'lidar' / f'{meta["frame"]}.laz'
    points = transform(np.asarray(laspy.read(path).xyz), meta['ego_to_lidar'])
    points = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    ax.set_facecolor('#101c2c')
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='winter',
               s=6, linewidths=0, alpha=.7)
    for corners, color, style in [(original, '#ffcc66', '--'),
                                   (shifted, '#ff604b', '-')]:
        for i, j in EDGES:
            ax.plot(corners[[i, j], 0], corners[[i, j], 1], color=color,
                    linestyle=style, linewidth=1.8)
        center = corners.mean(0)
        # CARLA vertex ordering: vertices 0..3 are the local +X (front) face.
        front = corners[:4].mean(0)
        direction = (front - center)[:2] * .7
        ax.scatter(*center[:2], c=color, s=28)
        ax.arrow(*center[:2], *direction, width=.025, head_width=.17,
                 length_includes_head=True, color=color)
    center0, center1 = original.mean(0), shifted.mean(0)
    ax.annotate('', xy=center1[:2], xytext=center0[:2],
                arrowprops=dict(arrowstyle='->', color='white', lw=1.6))
    ax.legend(handles=[Line2D([], [], color='#ffcc66', ls='--', label='Original box'),
                       Line2D([], [], color='#ff604b', label='Perturbed box')], loc='upper right')
    ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], xlabel='X forward (m)',
           ylabel='Y right (m)', aspect='equal')
    ax.grid(alpha=.15)
    ax.set_title(f'BEV | frame {meta["frame"]} | object {meta["box_id"]}\n'
                 f'dX = {dx:+.3f} m    dY = {dy:+.3f} m    '
                 f'dYaw = {yaw:+.2f} deg    seed = {seed}', fontsize=14, pad=16)
    fig.savefig(args.output / 'bev_comparison.png', dpi=180)
    plt.close(fig)
    # Store original annotation and perturbation separately for later correction planning.
    inverse_translation = np.array([-dx, -dy, 0])
    result = dict(frame=meta['frame'], box_id=meta['box_id'], route=meta['route'],
                  seed=seed, distribution='independent uniform per parameter',
                  max_xy_m=args.max_xy, max_yaw_deg=args.max_yaw_deg,
                  offset=dict(dx_m=float(dx), dy_m=float(dy), dyaw_deg=float(yaw)),
                  correction_about_perturbed_center=dict(
                      dx_m=float(inverse_translation[0]), dy_m=float(inverse_translation[1]),
                      dyaw_deg=float(-yaw)),
                  rotation='about box center, then translate; positive yaw rotates +X toward +Y',
                  coordinates=meta['lidar_coordinates'],
                  original_corners_lidar=original.tolist(),
                  perturbed_corners_lidar=shifted.tolist(),
                  original_center_lidar=center0.tolist(),
                  perturbed_center_lidar=center1.tolist())
    (args.output / 'offset.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.output / 'geometry.npz', points=points,
                        original_corners_lidar=original, perturbed_corners_lidar=shifted)
    print(json.dumps(result['offset'], indent=2))
    print(f'Seed: {seed}; saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
