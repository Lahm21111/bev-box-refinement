"""Choose and apply BEV box actions with the Gemini Decisions REST API."""
import argparse
import base64
import json
import math
import os
import re
import time
from pathlib import Path
import sys
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
from model_tools.actions import validate_actions, apply_action, GATE_ACTIONS, axis_options
from model_runtime.prompts import (VIEWS, INSTRUCTIONS, REFERENCE_INSTRUCTIONS,
                                  SAME_TARGET_INSTRUCTIONS, SAME_TARGET_REFERENCE_INSTRUCTIONS,
                                  GATE_INSTRUCTIONS, GATE_CRITERIA, axis_questions,
                                  view_descriptions, dataset_instructions, VEHICLE_ENCLOSURE_INSTRUCTIONS)

ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/decisions'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def create_run_directory(requested):
    """Reserve a fresh run directory without overwriting earlier attempts."""
    requested = Path(requested)
    index = 0
    while True:
        candidate = requested if index == 0 else requested.with_name(f'{requested.name}_{index:03d}')
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1


def image_blocks(images_dir, mapping, resolution, label):
    blocks = []
    for name, description in view_descriptions(mapping):
        path = Path(images_dir) / f'{name}.png'
        with Image.open(path) as image:
            if list(image.size) != mapping['views'][name]['size_wh']:
                raise ValueError(f'Image size does not match pixel mapping: {path}')
        blocks.extend([dict(type='text', text=f'{label} / {description}\n'),
                       dict(type='image', data=base64.b64encode(path.read_bytes()).decode('ascii'),
                            mime_type='image/png', resolution=resolution)])
    return blocks


def validate_same_target_reference(reference, current):
    if reference.get('view_frame','lidar') != current.get('view_frame','lidar'):
        raise ValueError('Same-target reference must use the same coordinate frame')
    if current.get('view_frame') == 'box' and not np.allclose(reference['view_to_lidar'], current['view_to_lidar']):
        raise ValueError('Same-target reference must use the current box frame')
    if any(reference[key] != current[key] for key in ['frame', 'box_id']):
        raise ValueError('Same-target reference must have the same frame and box ID')
    if not np.allclose(reference['bounds_lidar'], current['bounds_lidar']):
        raise ValueError('Same-target reference must use identical LiDAR crop bounds')
    for name, _ in VIEWS:
        ref_view, current_view = reference['views'][name], current['views'][name]
        keys = (['size_wh', 'lidar_axes', 'pixels_per_meter', 'u_min_m', 'v_max_m']
                if name.startswith('lidar') else ['size_wh', 'crop_xyxy'])
        if any(not np.allclose(ref_view[key], current_view[key]) for key in keys):
            raise ValueError(f'Same-target reference has mismatched pixel geometry: {name}')


def build_request(images_dir, vocabulary, mapping, resolution='ultra_high', reference_example=None,
                  reference_mode='example', stage='gate', pixel_step=None, yaw_step=None):
    if stage not in ['gate', 'adjustment']:
        raise ValueError('Unknown decision stage')
    if reference_mode not in ['example', 'same-target']:
        raise ValueError('Unknown reference mode')
    if reference_mode == 'same-target' and reference_example is None:
        raise ValueError('Same-target mode requires --reference-example')
    validate_actions(vocabulary, mapping)
    axis_options(pixel_step, yaw_step)
    instructions = SAME_TARGET_INSTRUCTIONS if reference_mode == 'same-target' else INSTRUCTIONS
    if stage == 'gate':
        instructions = GATE_INSTRUCTIONS.replace(
            '(5 or 10 original BEV pixels, 1 or 5 degrees)',
            f'({pixel_step if pixel_step is not None else "1, 5, or 10"} original BEV pixels, '
            f'{yaw_step if yaw_step is not None else "1, 2, or 5"} degrees)')
        questions = dict(next_action=dict(type='choice', instructions=instructions, criteria=dict(GATE_CRITERIA)))
    else:
        questions = axis_questions(pixel_step, yaw_step)
        for axis, question in questions.items():
            question['instructions'] = instructions + f'\nThis question controls only {axis}. Choose one criterion.'
    for question in questions.values():
        question['instructions'] = dataset_instructions(
            question['instructions'] + '\n\n' + VEHICLE_ENCLOSURE_INSTRUCTIONS, mapping)
    # Deliberately whitelist model-visible metadata: never send injected offsets,
    # correction labels, seed, original corners, or ground-truth annotation.
    view_info = {name: {key: value for key, value in mapping['views'][name].items()
                        if key in ['size_wh', 'lidar_axes', 'pixels_per_meter']}
                 for name, _ in VIEWS}
    context = dict(BEV_pixels_per_meter=mapping['views']['lidar_top']['pixels_per_meter'],
                   view_geometry=view_info)
    inputs = []
    if reference_example is not None:
        reference_mapping = read_json(Path(reference_example) / 'pixel_mapping.json')
        if reference_mode == 'same-target':
            validate_same_target_reference(reference_mapping, mapping)
        reference_text = (SAME_TARGET_REFERENCE_INSTRUCTIONS if reference_mode == 'same-target'
                          else REFERENCE_INSTRUCTIONS)
        inputs.append(dict(type='text', text=reference_text + '\n'))
        label = 'CORRECT REFERENCE - SAME TARGET' if reference_mode == 'same-target' else 'CORRECT REFERENCE EXAMPLE'
        inputs.extend(image_blocks(reference_example, reference_mapping, resolution, label))
    inputs.append(dict(type='text', text='CURRENT TARGET - choose an action for this vehicle only.\n'
                   'Current four-view geometry (original PNG pixels):\n'
                   + json.dumps(context, ensure_ascii=False) + '\n'))
    inputs.extend(image_blocks(images_dir, mapping, resolution, 'CURRENT TARGET'))
    return dict(model='gemini-decision-1', input=inputs, questions=questions)


def call_api(payload, api_key, timeout, max_retries=3):
    request = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json',
                                              'x-goog-api-key': api_key}, method='POST')
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            message = error.read().decode('utf-8', errors='replace')[:2000]
            if error.code != 429 or attempt == max_retries:
                code = error.code
                error.close()
                raise RuntimeError(f'Gemini HTTP {code}: {message}') from None
            retry_after = error.headers.get('Retry-After') if error.headers else None
            hint = re.search(r'retry in\s+(\d+(?:\.\d+)?)\s*s', message, re.IGNORECASE)
            try:
                delay = float(retry_after) if retry_after else (float(hint.group(1)) if hint else 60.)
                if not math.isfinite(delay) or delay < 0:
                    delay = 60.
            except ValueError:
                delay = float(hint.group(1)) if hint else 60.
            error.close()
            print(f'Gemini rate limit: waiting {delay:g}s before retry {attempt + 1}/{max_retries}.', flush=True)
            time.sleep(delay)
        except urllib.error.URLError as error:
            raise RuntimeError(f'Gemini connection failed: {error.reason}') from None


def parse_answer(response, by_id, question='next_action'):
    answer = response.get('answers', {}).get(question, {})
    if answer.get('type') != 'choice' or answer.get('choice') not in by_id:
        raise ValueError('API did not return a recognized choice answer')
    confidence = answer.get('confidence')
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError('Invalid decision confidence')
    probabilities = answer.get('probabilities', {})
    if set(probabilities) != set(by_id):
        raise ValueError('API probabilities do not cover the candidate vocabulary')
    if any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in probabilities.values()):
        raise ValueError('Invalid candidate probabilities')
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=.02):
        raise ValueError('Candidate probabilities are not normalized')
    ranking = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return dict(choice=answer['choice'], action=by_id[answer['choice']],
                confidence=confidence, top_candidates=ranking[:5])



def decide_round(images, vocabulary, mapping, folder, resolution='ultra_high',
                 reference_example=None, reference_mode='example', api_key=None,
                 timeout=120, dry_run=False, replay=None, pixel_step=None, yaw_step=None):
    """Run the three-way gate, calling movement selection only after refine."""
    validate_actions(vocabulary, mapping)
    gate_decision = None
    for stage in ['gate', 'adjustment']:
        stage_folder = folder / stage
        stage_folder.mkdir()
        payload = build_request(images, vocabulary, mapping, resolution,
                                reference_example, reference_mode, stage=stage,
                                pixel_step=pixel_step, yaw_step=yaw_step)
        write_json(stage_folder / 'request.json', payload)
        (stage_folder / 'instructions.txt').write_text(
            '\n\n'.join(q['instructions'] for q in payload['questions'].values()))
        if dry_run:
            print(f'Dry run {stage}: {sum(b["type"] == "image" for b in payload["input"])} images, '
                  f'{len(payload["questions"])} question(s), '
                  f'{[len(q["criteria"]) for q in payload["questions"].values()]} choices')
            continue
        if replay is not None:
            if stage not in replay:
                raise ValueError(f'Replay bundle is missing the {stage} API response')
            response = replay[stage]
        else:
            response = call_api(payload, api_key, timeout)
        write_json(stage_folder / 'response.json', response)
        if stage == 'gate':
            decision = parse_answer(response, GATE_ACTIONS)
        else:
            components = {}
            values = {}
            for axis, options in axis_options(pixel_step, yaw_step).items():
                components[axis] = parse_answer(response, options, question=axis)
                values[axis] = components[axis]['action']['value']
            action = dict(id='axis_adjustment', type='adjust', terminal=False,
                          dx_px=values['x'], dy_px=values['y'], dyaw_deg=values['yaw'],
                          dx_m=values['x'] / vocabulary['pixels_per_meter'],
                          dy_m=-values['y'] / vocabulary['pixels_per_meter'])
            if mapping.get('view_frame') == 'box':
                pose = np.asarray(mapping['view_to_lidar'])
                local_delta = np.array([action['dx_m'], action['dy_m'], 0.])
                delta = pose[:3,:3] @ local_delta
                action.update(coordinate_system='box', dx_box_m=float(local_delta[0]), dy_box_m=float(local_delta[1]),
                              dx_m=float(delta[0]), dy_m=float(delta[1]), dz_m=float(delta[2]),
                              translation_lidar_m=delta.tolist(),rotation_axis_lidar=pose[:3,2].tolist())
            if not any(values.values()):
                action.update(id='no_adjustment', type='stalled', terminal=True)
            decision = dict(choice=action['id'], action=action, component_decisions=components)
        write_json(stage_folder / 'decision.json', decision)
        if stage == 'gate':
            gate_decision = decision
            if decision['choice'] != 'refine_box':
                return dict(**decision, gate_decision=gate_decision)
        else:
            return dict(**decision, gate_decision=gate_decision)
    return None


def prepare_render_mapping(folder, state_path, source, images, reference_example=None,
                           reference_mode='example', data_root=None):
    """Expand current/reference windows together while preserving pixel units."""
    from model_tools.crop_mapping import expand_mapping, recover_corners
    from model_tools.render_views import render_views
    folder, source, images = Path(folder), Path(source), Path(images)
    state = read_json(state_path)
    mapping = read_json(images / 'pixel_mapping.json')
    meta = read_json(source / 'metadata.json')
    boxes = [np.asarray(state['perturbed_corners_lidar']), np.asarray(state['original_corners_lidar'])]
    if (folder / 'before_state.json').exists():
        boxes.append(np.asarray(read_json(folder / 'before_state.json')['perturbed_corners_lidar']))
    reference_corners = None
    if reference_mode == 'same-target':
        reference_mapping = read_json(Path(reference_example) / 'pixel_mapping.json')
        validate_same_target_reference(reference_mapping, mapping)
        reference_corners = recover_corners(reference_mapping)
        boxes.append(reference_corners)
    if mapping.get('view_frame') == 'box':
        from model_tools.box_views import make_box_mapping
        expanded = make_box_mapping(np.asarray(state['perturbed_corners_lidar']), meta, mapping, boxes[1:], data_root)
        changed = True
    else:
        expanded, changed = expand_mapping(mapping, np.concatenate(boxes), meta, data_root)
    if not changed:
        return images / 'pixel_mapping.json', reference_example
    mapping_path = folder / 'expanded_mapping.json'
    write_json(mapping_path, expanded)
    write_json(folder / 'crop_update.json', dict(
        old_bounds_lidar=mapping['bounds_lidar'], new_bounds_lidar=expanded['bounds_lidar'],
        old_camera_crop=mapping['views']['camera_crop']['crop_xyxy'],
        new_camera_crop=expanded['views']['camera_crop']['crop_xyxy'],
        reason='Expanded to include updated box; per-view pixel scales are unchanged'))
    if reference_corners is not None:
        reference_state = dict(state, perturbed_corners_lidar=reference_corners.tolist(),
            original_corners_lidar=reference_corners.tolist(), offset=dict(dx_m=0.,dy_m=0.,dyaw_deg=0.))
        write_json(folder / 'reference_state.json', reference_state)
        reference_example = folder / 'reference'
        render_views(source, folder / 'reference_state.json', data_root, reference_example,
                     reference_mapping=mapping_path)
    print('Updated box-local view frame and reference; pixel scales preserved.' if mapping.get('view_frame') == 'box'
          else 'Expanded view crops; pixel scales preserved and same-target reference synchronized.')
    return mapping_path, reference_example


def should_visualize(round_number, interval, max_rounds, terminal=False):
    return terminal or round_number == max_rounds or round_number % interval == 0


def write_comparison_gallery(output, history):
    rows = []
    for item in history:
        round_name = f'round_{item["round"]:03d}'
        if not (Path(output) / round_name / 'comparison').exists():
            continue
        if (Path(output) / round_name / 'comparison/before_after.png').exists():
            cells = (f'<td colspan="4"><a href="{round_name}/comparison/index.html"><img '
                     f'src="{round_name}/comparison/before_after.png"></a></td>')
        else:
            cells = ''.join(f'<td><a href="{round_name}/comparison/{name}.png"><img '
                            f'src="{round_name}/comparison/{name}.png"></a></td>' for name, _ in VIEWS)
        rows.append(f'<tr><th>{round_name}<br>{item["choice"]}</th>{cells}</tr>')
    (Path(output) / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Box refinement steps</title>'
        '<style>body{background:#17202b;color:white;font-family:sans-serif}'
        'table{width:100%;table-layout:fixed}th{width:12%}img{width:100%}</style>'
        '<p>Before/after panels show the actual input and updated images, each in its own box frame. '
        'Separate overlay images use a shared frame: yellow dashed is previous, red solid is updated.</p>'
        '<table><tr><th>Round / action</th><td>BEV XY</td><td>Front YZ</td>'
        '<td>Side XZ</td><td>Camera</td></tr>' + ''.join(rows) + '</table>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('outputs/small_offset_clean_views'))
    parser.add_argument('--actions', type=Path, default=Path('outputs/bev_small_offset/actions.json'))
    parser.add_argument('--state', type=Path, default=Path('outputs/bev_small_offset/offset.json'))
    parser.add_argument('--source', type=Path, default=Path('outputs/box_crops'))
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=Path('outputs/gemini_decision'))
    parser.add_argument('--max-rounds', type=int, default=1)
    parser.add_argument('--visualize-every', type=int, default=1,
                        help='Save before/after comparisons every N rounds and on the final round')
    parser.add_argument('--resolution', choices=['low', 'medium', 'high', 'ultra_high'], default='ultra_high')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--pixel-step', type=int, choices=[1, 5, 10], default=None)
    parser.add_argument('--yaw-step', type=int, choices=[1, 2, 5], default=None)
    parser.add_argument('--dry-run', action='store_true', help='Save a request for inspection without API calls')
    parser.add_argument('--response-file', type=Path, help='Replay one round from a JSON bundle with gate and optional adjustment API responses')
    parser.add_argument('--reference-example', type=Path,
                        help='Directory containing four correctly aligned example images and pixel_mapping.json')
    parser.add_argument('--reference-mode', choices=['example', 'same-target'], default='example',
                        help='Use a different-vehicle example or the exact same target with a correct box')
    args = parser.parse_args()
    if args.max_rounds < 1 or args.timeout <= 0 or args.visualize_every < 1:
        parser.error('max-rounds, visualize-every, and timeout must be positive')
    if args.response_file and args.max_rounds != 1:
        parser.error('--response-file supports one round only')
    if args.reference_mode == 'same-target' and args.reference_example is None:
        parser.error('--reference-mode same-target requires --reference-example')
    vocabulary = read_json(args.actions)
    state = read_json(args.state)
    images = args.images.resolve()
    mapping = read_json(images / 'pixel_mapping.json')
    if (state['frame'], state['box_id']) != (mapping['frame'], mapping['box_id']):
        parser.error('State and images belong to different samples')
    if state.get('status') in ['accepted', 'deleted']:
        parser.error('State is already terminal')
    by_id = validate_actions(vocabulary, mapping)
    # Ensure the current state actually generated these images, even for the same frame.
    c = np.asarray(state['perturbed_corners_lidar'])
    if mapping.get('view_frame') == 'box':
        from model_tools.geometry import transform
        c = transform(c, mapping['lidar_to_view'])
    for name, _ in VIEWS[:3]:
        view = mapping['views'][name]
        u, v = view['lidar_axes']
        projected = np.c_[(c[:, u] - view['u_min_m']) * view['pixels_per_meter'],
                          (view['v_max_m'] - c[:, v]) * view['pixels_per_meter']]
        if not np.allclose(projected, view['corners_pixels'], atol=1e-5):
            parser.error('Current state does not match rendered images; rerender first')
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not args.dry_run and not args.response_file and not api_key:
        parser.error('Set GEMINI_API_KEY or GOOGLE_API_KEY, or use --dry-run')
    requested_output = args.output
    args.output = create_run_directory(requested_output)
    if args.output != requested_output:
        print(f'Output already exists; using new run directory: {args.output.resolve()}')
    write_json(args.output / 'run.json', dict(
        mode='dry_run' if args.dry_run else ('response_replay' if args.response_file else 'live_api'),
        decision_flow='gate_then_independent_axes', pixel_step=args.pixel_step, yaw_step=args.yaw_step, model='gemini-decision-1', initial_images=str(images), actions=str(args.actions),
        response_file=str(args.response_file) if args.response_file else None,
        reference_example=str(args.reference_example.resolve()) if args.reference_example else None,
        reference_mode=args.reference_mode,
        max_rounds=args.max_rounds, visualize_every=args.visualize_every))
    history = []
    for index in range(args.max_rounds):
        folder = args.output / f'round_{index + 1:03d}'
        if folder.exists():
            parser.error(f'Output already exists: {folder}; choose a new --output directory')
        folder.mkdir()
        decision = decide_round(
            images, vocabulary, mapping, folder, args.resolution,
            args.reference_example, args.reference_mode, api_key, args.timeout,
            dry_run=args.dry_run,
            replay=read_json(args.response_file) if args.response_file else None,
            pixel_step=args.pixel_step, yaw_step=args.yaw_step)
        if args.dry_run:
            print(f'Gate and conditional adjustment requests saved to {folder.resolve()}')
            return
        write_json(folder / 'decision.json', decision)
        history.append(dict(round=index + 1, **decision))
        write_json(folder / 'before_state.json', state)
        state = apply_action(state, decision['action'], vocabulary['pixels_per_meter'], mapping=mapping)
        write_json(folder / 'state.json', state)
        write_json(args.output / 'history.json', history)
        write_json(args.output / 'final_state.json', state)
        print(f'Round {index + 1}: gate={decision["gate_decision"]["choice"]}; '
              f'action={decision["choice"]}; status={state["status"]}')
        if 'component_decisions' in decision:
            for axis, component in decision['component_decisions'].items():
                print(f'  {axis}: {component["action"]["value"]:+g}; '
                      f'confidence={component["confidence"]:.4f}')
        from model_tools.render_views import render_views
        render_mapping, args.reference_example = prepare_render_mapping(
            folder, folder / 'state.json', args.source, images,
            args.reference_example, args.reference_mode, args.data_root)
        if not decision['action']['terminal']:
            next_images = folder / 'views'
            render_views(source=args.source, data_root=args.data_root,
                         offset=folder / 'state.json', output=next_images,
                         reference_mapping=render_mapping)
        if should_visualize(index + 1, args.visualize_every, args.max_rounds, decision['action']['terminal']):
            comparison = folder / 'comparison'
            render_views(source=args.source, data_root=args.data_root,
                         offset=folder / 'state.json', output=comparison,
                         reference_mapping=render_mapping,
                         previous_offset=folder / 'before_state.json')
            from model_tools.comparison import save_actual_comparison
            save_actual_comparison(comparison, images,
                                   next_images if not decision['action']['terminal'] else images)
            write_comparison_gallery(args.output, history)
            print(f'Before/after comparison: {comparison.resolve()}')
        if decision['action']['terminal']:
            break
        images = next_images.resolve()
        mapping = read_json(images / 'pixel_mapping.json')
        print(f'Updated clean views: {images}')
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        sys.exit(f'Error: {error}')
