"""Generate ten four-view cases for accept/refine/delete evaluation."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from model_runtime.gemini_box_decision import create_run_directory, read_json, write_json
from model_tools.actions import build_actions
from model_tools.geometry import perturb_box, transform, project
from model_tools.render_views import render_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/status_examples_10'))
    parser.add_argument('--data-root', type=Path, default=Path('data/bench2drive'))
    args = parser.parse_args()
    targets = [
        ('nissan', Path('outputs/box_crops'), Path('outputs/medium_offset_clean_views/pixel_mapping.json')),
        ('impala', Path('outputs/examples/impala_00160_7477/source'),
         Path('outputs/examples/impala_00160_7477/views/pixel_mapping.json')),
    ]
    # Neutral case IDs; category labels are evaluator metadata, never model inputs.
    specs = [(0, 'large', 30, 20, 5), (1, 'correct', 0, 0, 0),
             (0, 'small', 5, -5, 1), (1, 'large', -30, -20, -5),
             (0, 'correct', 0, 0, 0), (1, 'small', -5, 5, -1),
             (0, 'large', 0, -30, 5), (1, 'small', 10, -5, 1),
             (0, 'small', -10, 5, -1), (1, 'large', 40, 0, 5)]
    output = create_run_directory(args.output)
    # Extend each shared camera crop to contain every planned variant. Keep the
    # original LiDAR bounds and scale, and use the same camera crop for all variants.
    crop_folder = output / 'crop_mappings'
    crop_folder.mkdir()
    prepared_targets = []
    for target, (name, source, reference) in enumerate(targets):
        meta, mapping = read_json(source / 'metadata.json'), read_json(reference)
        ppm = mapping['views']['lidar_top']['pixels_per_meter']
        with np.load(source / 'geometry.npz') as geometry:
            original = geometry['corners_lidar'].copy()
        pixels = []
        for t, _, x, y, yaw in specs:
            if t == target:
                corners = perturb_box(original, x / ppm, -y / ppm, yaw)
                world = transform(corners, np.linalg.inv(np.asarray(meta['sensors']['LIDAR_TOP']['world2lidar'])))
                projected, _ = project(world, meta['sensors'][meta['camera']])
                pixels.extend(projected)
        pixels = np.asarray(pixels)
        old_rect = mapping['views']['camera_crop']['crop_xyxy']
        camera_dir = 'rgb_' + meta['camera'].removeprefix('CAM_').lower()
        with Image.open(args.data_root / meta['route'] / 'camera' / camera_dir / f'{meta["frame"]}.jpg') as image:
            size = np.array(image.size)
        low = np.minimum(old_rect[:2], np.floor(pixels.min(0) - 12))
        high = np.maximum(old_rect[2:], np.ceil(pixels.max(0) + 12))
        rect = np.r_[np.maximum(low, 0), np.minimum(high, size)].astype(int)
        mapping['views']['camera_crop']['crop_xyxy'] = rect.tolist()
        mapping['views']['camera_crop']['size_wh'] = (rect[2:] - rect[:2]).tolist()
        reference = crop_folder / f'{name}.json'
        write_json(reference, mapping)
        prepared_targets.append((name, source, reference))
    targets = prepared_targets
    cases = []
    for index, (target, category, x, y, yaw) in enumerate(specs, 1):
        name, source, reference = targets[target]
        meta, mapping = read_json(source / 'metadata.json'), read_json(reference)
        ppm = mapping['views']['lidar_top']['pixels_per_meter']
        with np.load(source / 'geometry.npz') as geometry:
            original = geometry['corners_lidar'].copy()
        dx, dy = x / ppm, -y / ppm
        folder = output / f'case_{index:02d}'
        folder.mkdir()
        state = dict(route=meta['route'], frame=meta['frame'], box_id=meta['box_id'], seed=0,
                     offset=dict(dx_m=dx, dy_m=dy, dyaw_deg=yaw),
                     original_corners_lidar=original.tolist(),
                     perturbed_corners_lidar=perturb_box(original, dx, dy, yaw).tolist())
        write_json(folder / 'state.json', state)
        actions = build_actions(ppm)
        write_json(folder / 'actions.json', dict(count=len(actions), pixels_per_meter=ppm, actions=actions))
        rendered = render_views(source, folder / 'state.json', args.data_root, folder / 'views',
                                reference_mapping=reference)
        for view in rendered['views'].values():
            pixels = np.asarray(view['corners_pixels'])
            if np.any(pixels < 0) or np.any(pixels >= np.asarray(view['size_wh'])):
                raise ValueError(f'{folder.name}: box is clipped; choose another perturbation')
        cases.append(dict(id=folder.name, target=name, category=category,
                          expected_status='accept_box' if category == 'correct' else 'refine_box',
                          offset_bev_pixels=dict(dx_px=x, dy_px=y, dyaw_deg=yaw)))
        print(f'{folder.name}: {name}, {category}, BEV ({x}, {y}) px, yaw {yaw} deg')
    for case in cases:
        case['reference_case'] = next(c['id'] for c in cases
                                      if c['target'] == case['target'] and c['category'] == 'correct')
    write_json(output / 'manifest.json', dict(cases=cases,
        notes='Ten variants of two vehicles, not ten independent vehicles. Expected labels are synthetic '
              'test labels based on dataset annotations; small-offset acceptability depends on tolerance. '
              'No false-positive targets: delete behavior is not covered.'))
    rows = []
    for case in cases:
        images = ''.join(f'<td><a href="{case["id"]}/views/{name}.png"><img '
                         f'src="{case["id"]}/views/{name}.png"></a></td>'
                         for name in ['lidar_top', 'lidar_front', 'lidar_side', 'camera_crop'])
        rows.append(f'<tr><th>{case["id"]}</th>{images}</tr>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Status examples</title>'
        '<style>body{background:#17202b;color:white;font-family:sans-serif}table{width:100%;table-layout:fixed}'
        'td{vertical-align:middle}img{width:100%}th{width:7%}</style><table><tr><th>Case</th>'
        '<td>BEV XY</td><td>Front YZ</td><td>Side XZ</td><td>Camera</td></tr>' + ''.join(rows) + '</table>')
    print(f'Examples: {output.resolve()}')


if __name__ == '__main__':
    main()
