import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from model_tools.actions import build_actions, apply_action, GATE_ACTIONS, axis_options
from model_runtime.gemini_box_decision import build_request, parse_answer, create_run_directory, decide_round
from model_runtime.prompts import VIEWS


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.vocab = dict(count=125, pixels_per_meter=100., actions=build_actions(100.))
        self.by_id = {a['id']: a for a in self.vocab['actions']}

    def test_repeated_runs_preserve_existing_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / 'run'
            self.assertEqual(create_run_directory(requested), requested)
            marker = requested / 'run.json'
            marker.write_text('previous attempt')
            self.assertEqual(create_run_directory(requested), requested.with_name('run_001'))
            self.assertEqual(create_run_directory(requested), requested.with_name('run_002'))
            self.assertEqual(marker.read_text(), 'previous attempt')

    def test_request_contains_only_current_images_and_whitelisted_metadata(self):
        mapping = dict(offset='HIDDEN_GROUND_TRUTH', seed='HIDDEN_RANDOM_SEED', views={})
        with tempfile.TemporaryDirectory() as folder:
            for name, _ in VIEWS:
                Image.new('RGB', (64, 32)).save(Path(folder) / f'{name}.png')
                mapping['views'][name] = dict(size_wh=[64, 32], pixels_per_meter=100.,
                                              corners_pixels='HIDDEN_CORNERS')
            payload = build_request(Path(folder), self.vocab, mapping, stage='adjustment')
        self.assertEqual(payload['model'], 'gemini-decision-1')
        self.assertEqual(set(payload['questions']), {'x', 'y', 'yaw'})
        self.assertTrue(all(len(q['criteria']) == 7 for q in payload['questions'].values()))
        self.assertNotIn('accept_box', json.dumps(payload))
        self.assertEqual(sum(b['type'] == 'image' for b in payload['input']), 4)
        self.assertNotIn('HIDDEN_', json.dumps(payload))
        self.assertTrue(json.dumps(payload, ensure_ascii=False).isascii(),
                        'All instructions, image descriptions, and action criteria must be English ASCII')
        self.assertTrue(all(b['resolution'] == 'ultra_high' for b in payload['input'] if b['type'] == 'image'))

    def test_pixel_y_sign_rigid_geometry_and_exact_inverse(self):
        original = np.array([[x, y, z] for x in [-2., 2.] for y in [-1., 1.] for z in [0., 2.]])
        state = dict(perturbed_corners_lidar=original.tolist(),
                     offset=dict(dx_m=0., dy_m=0., dyaw_deg=0.))
        action = next(a for a in self.vocab['actions'] if a['type'] == 'adjust'
                      and (a['dx_px'], a['dy_px'], a['dyaw_deg']) == (5, 10, 1))
        moved = apply_action(state, action, 100.)
        points = np.asarray(moved['perturbed_corners_lidar'])
        np.testing.assert_allclose(points.mean(0) - original.mean(0), [.05, -.1, 0], atol=1e-12)
        np.testing.assert_allclose(points[:, 2], original[:, 2])
        np.testing.assert_allclose(np.linalg.norm(points[:, None] - points, axis=2),
                                   np.linalg.norm(original[:, None] - original, axis=2))
        inverse = next(a for a in self.vocab['actions'] if a['type'] == 'adjust'
                       and (a['dx_px'], a['dy_px'], a['dyaw_deg']) == (-5, -10, -1))
        restored = apply_action(moved, inverse, 100.)
        np.testing.assert_allclose(restored['perturbed_corners_lidar'], original, atol=1e-12)
        self.assertEqual(state['offset']['dx_m'], 0.)

    def test_correct_reference_is_separate_from_current_target(self):
        with tempfile.TemporaryDirectory() as folder:
            current, reference = Path(folder) / 'current', Path(folder) / 'reference'
            current.mkdir()
            reference.mkdir()
            mapping = dict(offset='HIDDEN_CURRENT_OFFSET', views={})
            for name, _ in VIEWS:
                Image.new('RGB', (64, 32), 'red').save(current / f'{name}.png')
                Image.new('RGB', (64, 32), 'green').save(reference / f'{name}.png')
                mapping['views'][name] = dict(size_wh=[64, 32], pixels_per_meter=100.)
            (reference / 'pixel_mapping.json').write_text(json.dumps(mapping))
            payload = build_request(current, self.vocab, mapping, reference_example=reference, stage='adjustment')
        self.assertEqual(sum(b['type'] == 'image' for b in payload['input']), 8)
        texts = '\n'.join(b['text'] for b in payload['input'] if b['type'] == 'text')
        self.assertLess(texts.index('CORRECT REFERENCE EXAMPLE'), texts.index('CURRENT TARGET - choose'))
        self.assertEqual(sum(b.get('text', '').startswith('CURRENT TARGET / Image')
                             for b in payload['input']), 4)
        self.assertEqual(set(payload['questions']), {'x', 'y', 'yaw'})
        self.assertTrue(all(len(q['criteria']) == 7 for q in payload['questions'].values()))
        self.assertNotIn('accept_box', json.dumps(payload))
        self.assertNotIn('HIDDEN_CURRENT_OFFSET', json.dumps(payload))
        self.assertTrue(json.dumps(payload, ensure_ascii=False).isascii())

    def test_gate_terminals_preserve_geometry_and_nonzero_adjustments(self):
        state = dict(perturbed_corners_lidar=[[0, 0, 0]])
        self.assertNotIn('accept_box', self.by_id)
        self.assertEqual(sum(a['type'] == 'adjust' for a in self.vocab['actions']), 124)
        self.assertNotIn('action_062', self.by_id)
        self.assertFalse(any(a['type'] == 'adjust' and a['dx_px'] == a['dy_px'] == a['dyaw_deg'] == 0
                             for a in self.vocab['actions']))
        self.assertEqual([a['id'] for a in self.vocab['actions'] if a['terminal']], ['delete_box'])
        result = apply_action(state, self.by_id['delete_box'], 100.)
        self.assertEqual(result['status'], 'deleted')
        self.assertEqual(result['perturbed_corners_lidar'], state['perturbed_corners_lidar'])
        accepted = apply_action(state, GATE_ACTIONS['accept_box'], 100.)
        self.assertEqual(accepted['status'], 'accepted')
        self.assertEqual(accepted['perturbed_corners_lidar'], state['perturbed_corners_lidar'])

    def test_same_target_reference_uses_matching_frame_and_crop(self):
        mapping = dict(frame='00100', box_id='34598', offset='HIDDEN_OFFSET',
                       bounds_lidar=[[0, 0, 0], [1, 1, 1]], views={})
        with tempfile.TemporaryDirectory() as folder:
            current, reference = Path(folder) / 'current', Path(folder) / 'reference'
            current.mkdir()
            reference.mkdir()
            for name, _ in VIEWS:
                Image.new('RGB', (64, 32)).save(current / f'{name}.png')
                Image.new('RGB', (64, 32)).save(reference / f'{name}.png')
                mapping['views'][name] = dict(size_wh=[64, 32], pixels_per_meter=100.,
                                              lidar_axes=[0, 1], u_min_m=0., v_max_m=1.,
                                              crop_xyxy=[0, 0, 64, 32])
            ref_path = reference / 'pixel_mapping.json'
            ref_path.write_text(json.dumps(mapping))
            payload = build_request(current, self.vocab, mapping, reference_example=reference,
                                    reference_mode='same-target')
            self.assertEqual(sum(b['type'] == 'image' for b in payload['input']), 8)
            text = json.dumps(payload, ensure_ascii=False)
            self.assertIn('SAME VEHICLE IN THE SAME FRAME', text)
            self.assertNotIn('different vehicle', text)
            self.assertNotIn('HIDDEN_OFFSET', text)
            self.assertTrue(text.isascii())
            wrong = json.loads(json.dumps(mapping))
            wrong['frame'] = '00160'
            ref_path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, 'same frame'):
                build_request(current, self.vocab, mapping, reference_example=reference,
                              reference_mode='same-target')
            wrong['frame'] = '00100'
            wrong['views']['lidar_top']['pixels_per_meter'] = 200.
            ref_path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, 'mismatched pixel geometry'):
                build_request(current, self.vocab, mapping, reference_example=reference,
                              reference_mode='same-target')

    def test_two_stage_branching_and_candidate_separation(self):
        def response(candidates, choice):
            return dict(answers=dict(next_action=dict(type='choice', choice=choice,
                confidence=.8, probabilities={key: float(key == choice) for key in candidates})))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = dict(views={})
            for name, _ in VIEWS:
                Image.new('RGB', (64, 32)).save(root / f'{name}.png')
                mapping['views'][name] = dict(size_wh=[64, 32], pixels_per_meter=100.)
            for gate_choice in GATE_ACTIONS:
                folder = root / gate_choice
                folder.mkdir()
                responses = [response(GATE_ACTIONS, gate_choice)]
                if gate_choice == 'refine_box':
                    responses.append(dict(answers={axis: dict(type='choice', choice=choice, confidence=.8,
                        probabilities={k: float(k == choice) for k in axis_options()[axis]})
                        for axis, choice in [('x','negative_5'),('y','zero'),('yaw','positive_1')]}))
                with patch('model_runtime.gemini_box_decision.call_api', side_effect=responses) as api:
                    decision = decide_round(root, self.vocab, mapping, folder)
                self.assertEqual(api.call_count, 2 if gate_choice == 'refine_box' else 1)
                gate_payload = json.loads((folder / 'gate/request.json').read_text())
                self.assertEqual(set(gate_payload['questions']['next_action']['criteria']), set(GATE_ACTIONS))
                self.assertTrue(json.dumps(gate_payload, ensure_ascii=False).isascii())
                self.assertEqual(decision['gate_decision']['choice'], gate_choice)
                if gate_choice == 'refine_box':
                    payload = json.loads((folder / 'adjustment/request.json').read_text())
                    self.assertEqual(set(payload['questions']), {'x','y','yaw'})
                    for axis, q in payload['questions'].items():
                        self.assertEqual(set(q['criteria']), set(axis_options()[axis]))
                    self.assertEqual(decision['choice'], 'axis_adjustment')
                    self.assertEqual([decision['action'][k] for k in ['dx_px','dy_px','dyaw_deg']], [-5,0,1])
                else:
                    self.assertFalse((folder / 'adjustment').exists())
                    self.assertEqual(decision['choice'], gate_choice)
            replay_folder = root / 'replay'
            replay_folder.mkdir()
            with patch('model_runtime.gemini_box_decision.call_api') as api:
                result = decide_round(root, self.vocab, mapping, replay_folder,
                                      replay={'gate': response(GATE_ACTIONS, 'accept_box')})
            api.assert_not_called()
            self.assertEqual(result['choice'], 'accept_box')

    def test_independent_axis_steps_and_stalled_result(self):
        gate = dict(answers=dict(next_action=dict(type='choice', choice='refine_box', confidence=.9,
            probabilities={k: float(k == 'refine_box') for k in GATE_ACTIONS})))
        def adjustments(choices):
            return dict(answers={axis: dict(type='choice', choice=choice, confidence=.7,
                probabilities={k: float(k == choice) for k in ['negative', 'zero', 'positive']})
                for axis, choice in choices.items()})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = dict(views={})
            for name, _ in VIEWS:
                Image.new('RGB', (64, 32)).save(root / f'{name}.png')
                mapping['views'][name] = dict(size_wh=[64, 32], pixels_per_meter=100.)
            for name, choices in [('move', dict(x='positive', y='negative', yaw='negative')),
                                  ('stall', dict(x='zero', y='zero', yaw='zero'))]:
                folder = root / name
                folder.mkdir()
                decision = decide_round(root, self.vocab, mapping, folder, pixel_step=10, yaw_step=5,
                    replay=dict(gate=gate, adjustment=adjustments(choices)))
                state = dict(perturbed_corners_lidar=[[0,0,0]], offset=dict(dx_m=0.,dy_m=0.,dyaw_deg=0.))
                updated = apply_action(state, decision['action'], 100.)
                if name == 'stall':
                    self.assertEqual(updated['status'], 'stalled')
                    self.assertEqual(updated['perturbed_corners_lidar'], state['perturbed_corners_lidar'])
                    self.assertTrue(decision['action']['terminal'])
                else:
                    np.testing.assert_allclose(updated['perturbed_corners_lidar'], [[.1,.1,0]])
                    self.assertEqual(updated['offset']['dyaw_deg'], -5)
                payload = json.loads((folder / 'adjustment/request.json').read_text())
                self.assertIn('10 original BEV pixels', payload['questions']['x']['criteria']['positive'])
                self.assertIn('5 degrees', payload['questions']['yaw']['criteria']['positive'])
            folder = root / 'missing-axis'
            folder.mkdir()
            bad = adjustments(dict(x='positive',y='zero'))
            with self.assertRaises(ValueError):
                decide_round(root, self.vocab, mapping, folder, replay=dict(gate=gate,adjustment=bad))

    def test_comparison_keeps_projection_and_clean_model_image_separate(self):
        from model_tools.render_views import render_lidar
        current = np.array([[x,y,z] for x in [0.,2.] for y in [0.,1.] for z in [0.,1.]])
        previous = current + [0.2,0.2,0.]
        points = np.array([[.5,.5,.5]])
        bounds = np.array([[-1.,-1.,-1.],[3.,3.,2.]])
        clean, clean_mapping = render_lidar(points, current, bounds, [0,1], 128)
        comparison, comparison_mapping = render_lidar(points, current, bounds, [0,1], 128, previous)
        self.assertEqual(clean_mapping, comparison_mapping)
        a,b = np.asarray(clean),np.asarray(comparison)
        self.assertEqual(a.shape,b.shape)
        self.assertFalse(np.array_equal(a,b))
        yellow = lambda pixels: ((pixels[:,:,0]>180)&(pixels[:,:,1]>140)&(pixels[:,:,2]<130)).sum()
        self.assertEqual(yellow(a),0)
        self.assertGreater(yellow(b),0)

    def test_visualization_interval_and_final_snapshot(self):
        from model_runtime.gemini_box_decision import should_visualize, write_comparison_gallery
        self.assertEqual([n for n in range(1,101) if should_visualize(n,5,100)],list(range(5,101,5)))
        self.assertTrue(should_visualize(3,5,100,terminal=True))
        self.assertTrue(should_visualize(7,5,7))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'round_005/comparison').mkdir(parents=True)
            write_comparison_gallery(root,[dict(round=n,choice='axis_adjustment') for n in range(1,6)])
            html=(root/'index.html').read_text()
            self.assertIn('round_005/comparison',html)
            self.assertNotIn('round_001/comparison',html)

    def test_default_axis_ranges(self):
        options = axis_options()
        for axis, expected in [('x', {-10,-5,-1,0,1,5,10}),
                               ('y', {-10,-5,-1,0,1,5,10}),
                               ('yaw', {-5,-2,-1,0,1,2,5})]:
            self.assertEqual({v['value'] for v in options[axis].values()}, expected)
        self.assertEqual(options['x']['positive_1']['value'], 1)
        self.assertEqual(options['yaw']['negative_2']['value'], -2)

    def test_box_input_validation_and_offset_origin(self):
        from model_runtime.box_refinement_pipeline import prepare_box_input
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'metadata.json').write_text(json.dumps(dict(route='route',frame='00100',box_id='42')))
            corners = [[x,y,z] for x in [0,2] for y in [0,1] for z in [0,1]]
            box = root / 'box.json'
            box.write_text(json.dumps(dict(corners_lidar=corners)))
            mapping = dict(views=dict(lidar_top=dict(pixels_per_meter=100.)))
            with patch('model_runtime.box_refinement_pipeline.render_views', return_value=mapping):
                prepared = prepare_box_input(box, source, root, root / 'input')
            state = json.loads((prepared / 'state.json').read_text())
            self.assertEqual(state['perturbed_corners_lidar'], corners)
            self.assertEqual(state['offset'], dict(dx_m=0.,dy_m=0.,dyaw_deg=0.))
            self.assertIn('supplied input box', state['offset_origin'])
            for invalid in [[[0,0,0]], dict(corners_lidar=corners,box_id='wrong')]:
                box.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    prepare_box_input(box, source, root, root / 'invalid')
                self.assertFalse((root / 'invalid').exists())

    def test_parse_validates_vocabulary_and_probabilities(self):
        answer = dict(type='choice', choice='delete_box', confidence=.9,
                      probabilities={key: float(key == 'delete_box') for key in self.by_id})
        response = dict(answers=dict(next_action=answer))
        self.assertEqual(parse_answer(response, self.by_id)['choice'], 'delete_box')
        answer['choice'] = 'unknown_action'
        with self.assertRaises(ValueError):
            parse_answer(response, self.by_id)
        answer['choice'] = 'delete_box'
        answer['probabilities']['delete_box'] = .5
        with self.assertRaises(ValueError):
            parse_answer(response, self.by_id)


if __name__ == '__main__':
    unittest.main()
