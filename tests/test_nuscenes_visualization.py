import unittest
import tempfile
from pathlib import Path
from PIL import Image

import numpy as np

from data_pipeline.visualize_nuscenes_box import pose_matrix, box_corners, camera_project
from model_tools.geometry import transform


class NuScenesGeometryTests(unittest.TestCase):
    def test_quaternion_order_and_pose_inverse(self):
        pose = pose_matrix(dict(rotation=[np.sqrt(.5),0,0,np.sqrt(.5)], translation=[3,4,5]))
        moved = transform(np.array([[1.,0.,0.]]), pose)
        np.testing.assert_allclose(moved, [[3,5,5]])
        np.testing.assert_allclose(transform(moved, np.linalg.inv(pose)), [[1,0,0]], atol=1e-12)

    def test_box_size_is_width_length_height(self):
        corners = box_corners(dict(size=[2,4,1], rotation=[1,0,0,0], translation=[3,4,5]))
        np.testing.assert_allclose(np.ptp(corners,axis=0), [4,2,1])
        np.testing.assert_allclose(corners.mean(0), [3,4,5])

    def test_camera_uses_optical_axes_without_carla_axis_swap(self):
        intrinsic = [[100,0,50],[0,100,40],[0,0,1]]
        pixels, depth = camera_project(np.array([[1.,2.,10.]]), np.eye(4), intrinsic)
        np.testing.assert_allclose(pixels, [[60,60]])
        np.testing.assert_allclose(depth, [10])

    def test_nuscenes_model_inputs_use_native_lidar_axes(self):
        from model_runtime.gemini_box_decision import build_request
        from model_tools.actions import build_actions
        from model_runtime.prompts import VIEWS
        mapping = dict(dataset='nuScenes', offset='HIDDEN_OFFSET', views={})
        vocabulary = dict(count=125,pixels_per_meter=100.,actions=build_actions(100.))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,_ in VIEWS:
                Image.new('RGB',(64,32)).save(root/f'{name}.png')
                mapping['views'][name] = dict(size_wh=[64,32],pixels_per_meter=100.)
            for stage in ['gate','adjustment']:
                payload = build_request(root,vocabulary,mapping,stage=stage)
                texts=' '.join(b.get('text','') for b in payload['input'])
                self.assertIn('front view, XZ plane',texts)
                self.assertIn('side view, YZ plane',texts)
                self.assertNotIn('HIDDEN_OFFSET',texts)
                for question in payload['questions'].values():
                    self.assertNotIn('LiDAR X points forward',question['instructions'])
                if stage=='adjustment':
                    self.assertIn('nuScenes LiDAR X points right',payload['questions']['x']['instructions'])
                else:
                    self.assertIn('Image 2 is front XZ',payload['questions']['next_action']['instructions'])

    def test_fixed_reference_keeps_exact_scale_and_size(self):
        from model_tools.render_views import render_lidar
        corners=np.array([[x,y,z] for x in [0.,1.] for y in [0.,2.] for z in [0.,1.]])
        bounds=np.array([[-.2,-.4,-.2],[1.2,2.4,1.2]])
        points=np.array([[.1,.2,.3]])
        original,mapping=render_lidar(points,corners,bounds,[0,1],1024)
        repeated,new_mapping=render_lidar(points,corners,bounds,[0,1],max(mapping['size_wh'])+1,
                                         reference_view=mapping)
        self.assertEqual(mapping,new_mapping)
        np.testing.assert_array_equal(np.array(original),np.array(repeated))
