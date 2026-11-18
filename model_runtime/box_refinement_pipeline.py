"""Render one LiDAR-frame box and run gated, independent-axis refinement."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

from model_runtime import gemini_box_decision as runtime
from model_tools.actions import build_actions
from model_tools.nuscenes_views import sample_identity
from model_tools.render_views import render_views


def prepare_box_input(box_path, source, data_root, output, reference_mapping=None, view_frame='lidar'):
    """Accept eight ordered LiDAR corners, or an existing state containing them."""
    raw = runtime.read_json(box_path)
    meta = runtime.read_json(Path(source) / 'metadata.json')
    identity = sample_identity(meta)
    if isinstance(raw, dict):
        for key in ['route', 'frame', 'box_id']:
            if key in raw and raw[key] != identity[key]:
                raise ValueError(f'Input box and source disagree on {key}')
        raw = raw.get('perturbed_corners_lidar', raw.get('corners_lidar'))
    corners = np.asarray(raw, dtype=float)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError('Box must contain eight finite LiDAR-frame XYZ corners in dataset corner order')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = dict(**identity, seed=0,
                 original_corners_lidar=corners.tolist(), perturbed_corners_lidar=corners.tolist(),
                 offset=dict(dx_m=0., dy_m=0., dyaw_deg=0.),
                 offset_origin='Cumulative changes relative to the supplied input box')
    runtime.write_json(output / 'state.json', state)
    if view_frame == 'box':
        from model_tools.box_views import make_box_mapping
        from model_tools.crop_mapping import recover_corners
        reference = runtime.read_json(reference_mapping) if reference_mapping else None
        extras = [recover_corners(reference)] if reference else []
        frame_mapping = make_box_mapping(corners, meta, reference, extras, data_root)
        reference_mapping = output / 'view_mapping.json'
        runtime.write_json(reference_mapping, frame_mapping)
    mapping = render_views(source, output / 'state.json', data_root, output / 'views',
                           reference_mapping=reference_mapping, view_frame=view_frame)
    ppm = mapping['views']['lidar_top']['pixels_per_meter']
    actions = build_actions(ppm)
    runtime.write_json(output / 'actions.json', dict(count=len(actions), pixels_per_meter=ppm, actions=actions))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--box', type=Path, required=True,
                        help='JSON: eight LiDAR XYZ corners, corners_lidar, or a prior state JSON')
    parser.add_argument('--source', type=Path, default=Path('outputs/box_crops'),
                        help='Scene metadata/geometry directory matching this box and frame')
    parser.add_argument('--view-frame', choices=['box','lidar'], default='box',
                        help='Three LiDAR projections use current box XYZ by default; lidar keeps legacy fixed axes')
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--reference-mapping', type=Path,
                        help='Optional fixed four-view crop, required to match a same-target reference')
    parser.add_argument('--reference-example', type=Path)
    parser.add_argument('--reference-mode', choices=['example', 'same-target'], default='example')
    parser.add_argument('--output', type=Path, default=Path('outputs/box_pipeline'))
    parser.add_argument('--max-rounds', type=int, default=10)
    parser.add_argument('--visualize-every', type=int, default=1,
                        help='Save before/after comparisons every N rounds and on the final round')
    parser.add_argument('--pixel-step', type=int, choices=[1, 5, 10], default=None)
    parser.add_argument('--yaw-step', type=int, choices=[1, 2, 5], default=None)
    parser.add_argument('--resolution', choices=['low', 'medium', 'high', 'ultra_high'], default='ultra_high')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.max_rounds < 1 or args.timeout <= 0 or args.visualize_every < 1:
        parser.error('--max-rounds, --visualize-every, and --timeout must be positive')
    if args.reference_mode == 'same-target' and not args.reference_example:
        parser.error('--reference-mode same-target requires --reference-example')
    if not args.dry_run and not (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')):
        parser.error('Set GEMINI_API_KEY or GOOGLE_API_KEY, or use --dry-run')
    reference_mapping = args.reference_mapping
    if args.reference_mode == 'same-target' and reference_mapping is None:
        reference_mapping = args.reference_example / 'pixel_mapping.json'
    output = runtime.create_run_directory(args.output)
    prepared = prepare_box_input(args.box, args.source, args.data_root, output / 'input', reference_mapping, args.view_frame)
    if args.view_frame == 'box' and args.reference_mode == 'same-target':
        from model_tools.crop_mapping import recover_corners
        reference_corners = recover_corners(runtime.read_json(args.reference_example / 'pixel_mapping.json'))
        reference_state = runtime.read_json(prepared / 'state.json')
        reference_state.update(perturbed_corners_lidar=reference_corners.tolist(),
                               original_corners_lidar=reference_corners.tolist())
        runtime.write_json(prepared / 'reference_state.json', reference_state)
        render_views(args.source, prepared / 'reference_state.json', args.data_root, prepared / 'reference',
                     reference_mapping=prepared / 'views/pixel_mapping.json', view_frame='box')
        args.reference_example = prepared / 'reference'
    runtime.write_json(output / 'pipeline.json', dict(box=str(args.box.resolve()),
        source=str(args.source.resolve()), view_frame=args.view_frame, pixel_step=args.pixel_step, yaw_step=args.yaw_step,
        result_directory='run', max_rounds=args.max_rounds, visualize_every=args.visualize_every))
    arguments = ['gemini_box_decision', '--images', str(prepared / 'views'),
        '--state', str(prepared / 'state.json'), '--actions', str(prepared / 'actions.json'),
        '--source', str(args.source), '--output', str(output / 'run'),
        '--max-rounds', str(args.max_rounds), '--visualize-every', str(args.visualize_every), '--resolution', args.resolution, '--timeout', str(args.timeout),
        '--reference-mode', args.reference_mode]
    if args.data_root is not None:
        arguments.extend(['--data-root', str(args.data_root)])
    if args.pixel_step is not None:
        arguments.extend(['--pixel-step', str(args.pixel_step)])
    if args.yaw_step is not None:
        arguments.extend(['--yaw-step', str(args.yaw_step)])
    if args.reference_example:
        arguments.extend(['--reference-example', str(args.reference_example)])
    if args.dry_run:
        arguments.append('--dry-run')
    previous_argv = sys.argv
    try:
        sys.argv = arguments
        runtime.main()
    finally:
        sys.argv = previous_argv


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        sys.exit(f'Error: {error}')
