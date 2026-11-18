"""Action vocabulary, validation, and execution for box refinement."""
import copy
import itertools
import math

import numpy as np
from .geometry import perturb_box, perturb_box_local

GATE_ACTIONS = {
    'accept_box': dict(id='accept_box', type='accept', terminal=True),
    'refine_box': dict(id='refine_box', type='refine', terminal=False),
    'delete_box': dict(id='delete_box', type='delete', terminal=True),
}

def axis_options(pixel_step=None, yaw_step=None):
    """Independent signed increments; optional single-step legacy overrides."""
    if pixel_step not in [None, 1, 5, 10] or yaw_step not in [None, 1, 2, 5]:
        raise ValueError('Unsupported pixel or yaw step')
    result = {}
    for axis, override, defaults in [('x', pixel_step, [1, 5, 10]),
                                      ('y', pixel_step, [1, 5, 10]),
                                      ('yaw', yaw_step, [1, 2, 5])]:
        steps = [override] if override is not None else defaults
        result[axis] = {'zero': dict(value=0)}
        for step in steps:
            for sign, multiplier in [('negative', -1), ('positive', 1)]:
                key = sign if override is not None else f'{sign}_{step}'
                result[axis][key] = dict(value=multiplier * step)
    return result


def build_actions(ppm):
    steps, angles = [-10, -5, 0, 5, 10], [-5, -1, 0, 1, 5]
    actions = [dict(id=f'action_{i:03d}', type='adjust', terminal=False,
                    dx_px=x, dy_px=y, dyaw_deg=a, dx_m=x / ppm, dy_m=-y / ppm)
               for i, (x, y, a) in enumerate(itertools.product(steps, steps, angles))
               if (x, y, a) != (0, 0, 0)]
    actions.extend([
        dict(id='delete_box', type='delete', terminal=True,
             description='This box should not be kept as a valid target annotation. Delete it and end adjustment.'),
    ])
    return actions


def validate_actions(vocabulary, mapping):
    actions = vocabulary['actions']
    if vocabulary['count'] != len(actions) or len(actions) != 125:
        raise ValueError('Expected 125 action candidates: 124 nonzero adjustments and delete_box')
    by_id = {action['id']: action for action in actions}
    if len(by_id) != len(actions):
        raise ValueError('Duplicate action IDs')
    scale = mapping['views']['lidar_top']['pixels_per_meter']
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid BEV pixel scale')
    if not math.isclose(vocabulary['pixels_per_meter'], scale, rel_tol=1e-9):
        raise ValueError('Action pixel scale differs from input BEV image')
    for action in actions:
        if action['type'] == 'adjust':
            if action['dx_px'] == action['dy_px'] == action['dyaw_deg'] == 0:
                raise ValueError('All-zero adjustment is not allowed')
            if (action['dx_px'] not in [-10, -5, 0, 5, 10]
                    or action['dy_px'] not in [-10, -5, 0, 5, 10]
                    or action['dyaw_deg'] not in [-5, -1, 0, 1, 5]):
                raise ValueError('Unexpected adjustment step')
            if not np.allclose([action['dx_m'], action['dy_m']],
                               [action['dx_px'] / scale, -action['dy_px'] / scale]):
                raise ValueError('Inconsistent pixel-to-meter conversion')
        elif action['id'] != 'delete_box' or action['type'] != 'delete' or not action['terminal']:
            raise ValueError('Unknown terminal action')
    if set(by_id) != {a['id'] for a in build_actions(scale)}:
        raise ValueError('Action IDs must contain 124 nonzero adjustments and delete_box only')
    return by_id


def apply_action(state, action, scale, mapping=None):
    updated = copy.deepcopy(state)
    executed = copy.deepcopy(action)
    if action['type'] == 'adjust':
        if action['dx_px'] == action['dy_px'] == action['dyaw_deg'] == 0:
            raise ValueError('All-zero adjustment is not allowed')
        dx, dy, yaw = action['dx_px'] / scale, -action['dy_px'] / scale, action['dyaw_deg']
        if mapping is not None and mapping.get('view_frame') == 'box':
            pose = np.asarray(mapping['view_to_lidar'])
            corners = perturb_box_local(np.asarray(state['perturbed_corners_lidar']), dx, dy, yaw, pose)
            delta = pose[:3,:3] @ [dx,dy,0]
            executed.update(coordinate_system='box',translation_lidar_m=delta.tolist(),rotation_axis_lidar=pose[:3,2].tolist())
            dx,dy = delta[:2]
            if 'dz_m' not in updated['offset']:
                updated['offset']['dz_m'] = 0.
            updated['offset']['dz_m'] += float(delta[2])
        else:
            corners = perturb_box(np.asarray(state['perturbed_corners_lidar']), dx, dy, yaw)
        updated['perturbed_corners_lidar'] = corners.tolist()
        for key, value in [('dx_m', dx), ('dy_m', dy), ('dyaw_deg', yaw)]:
            updated['offset'][key] += value
        # Initial perturbation labels no longer describe the current geometry.
        for key in ['offset_bev_pixels', 'correction_bev_pixels', 'correction_about_perturbed_center']:
            updated.pop(key, None)
        updated['status'] = 'adjusting'
    elif action['type'] == 'stalled' and action['id'] == 'no_adjustment':
        updated['status'] = 'stalled'
    elif action['type'] == 'accept' and action['id'] == 'accept_box':
        updated['status'] = 'accepted'
    elif action['type'] == 'delete' and action['id'] == 'delete_box':
        updated['status'] = 'deleted'
    else:
        raise ValueError('Unsupported action type')
    updated.setdefault('applied_actions', []).append(executed)
    return updated
