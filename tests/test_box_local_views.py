import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from model_tools.geometry import box_pose, transform, perturb_box
from model_tools.actions import apply_action
from model_tools.box_views import make_box_mapping
from model_tools.crop_mapping import recover_corners
from model_tools.render_views import render_views
from model_runtime.gemini_box_decision import prepare_render_mapping, validate_same_target_reference, build_request
from model_tools.actions import build_actions


class BoxLocalViewTests(unittest.TestCase):
    def setUp(self):
        self.local=np.array([[x,y,z] for x in [2.,-2.] for y in [1.,-1.] for z in [1.,-1.]])

    def test_box_axes_recover_length_width_height_with_pitch_and_yaw(self):
        angle=np.deg2rad(40)
        rz=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        pitch=np.deg2rad(15)
        ry=np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        basis=rz@ry
        corners=self.local@basis.T+[4,3,10]
        pose=box_pose(corners)
        np.testing.assert_allclose(pose[:3,:3],basis,atol=1e-12)
        np.testing.assert_allclose(transform(corners,np.linalg.inv(pose)),self.local,atol=1e-12)

    def test_view_pixel_translation_is_rotated_back_to_sensor_frame(self):
        # Local +X points along sensor +Y. A rightward 10px move must not move sensor X.
        corners=perturb_box(self.local,0,0,90)+[0,0,10]
        pose=box_pose(corners)
        state=dict(perturbed_corners_lidar=corners.tolist(),offset=dict(dx_m=0.,dy_m=0.,dyaw_deg=0.))
        action=dict(id='axis_adjustment',type='adjust',terminal=False,dx_px=10,dy_px=5,dyaw_deg=2)
        moved=apply_action(state,action,100.,mapping=dict(view_frame='box',view_to_lidar=pose.tolist()))
        result=np.array(moved['perturbed_corners_lidar'])
        np.testing.assert_allclose(result.mean(0)-corners.mean(0),[.05,.1,0],atol=1e-12)
        np.testing.assert_allclose(transform(result,np.linalg.inv(pose)).mean(0),[.1,-.05,0],atol=1e-12)
        relative=pose[:3,:3].T@box_pose(result)[:3,:3]
        self.assertAlmostEqual(np.degrees(np.arctan2(relative[1,0],relative[0,0])),2.)
        np.testing.assert_allclose(np.linalg.norm(result[:,None]-result,axis=2),np.linalg.norm(corners[:,None]-corners,axis=2))

    def test_reference_and_target_follow_the_same_updated_box_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            Image.new('RGB',(160,120)).save(root/'camera.png')
            np.array([[0,0,10,1,0],[.2,.2,10,1,0]],dtype=np.float32).tofile(root/'lidar.bin')
            meta=dict(dataset='nuScenes',scene='test',sample_token='frame',box_token='box',data_root=str(root),
                lidar_filename='lidar.bin',camera_filename='camera.png',lidar_to_global=np.eye(4).tolist(),
                global_to_camera=np.eye(4).tolist(),camera_intrinsic=[[100,0,80],[0,100,60],[0,0,1]])
            (source/'metadata.json').write_text(json.dumps(meta))
            correct=self.local+[0,0,10]
            current=perturb_box(correct,.1,.1,12)
            identity=dict(dataset='nuScenes',route='nuScenes/test',frame='frame',box_id='box',seed=0,
                          offset=dict(dx_m=0.,dy_m=0.,dyaw_deg=0.))
            def state(c):return dict(identity,original_corners_lidar=c.tolist(),perturbed_corners_lidar=c.tolist())
            state_path=root/'state.json';state_path.write_text(json.dumps(state(current)))
            shared=make_box_mapping(current,meta,extra_corners=[correct],max_size=128)
            (root/'shared.json').write_text(json.dumps(shared))
            mapping=render_views(source,state_path,None,root/'views',max_size=128,view_frame='box',reference_mapping=root/'shared.json')
            reference_state=root/'reference_state.json';reference_state.write_text(json.dumps(state(correct)))
            ref=render_views(source,reference_state,None,root/'reference',reference_mapping=root/'views/pixel_mapping.json')
            validate_same_target_reference(ref,mapping)
            np.testing.assert_allclose(recover_corners(ref),correct,atol=1e-12)
            np.testing.assert_allclose(recover_corners(mapping),current,atol=1e-12)
            # The current box is axis aligned; the differently oriented reference is not.
            current_xy=np.array(mapping['views']['lidar_top']['corners_pixels'])
            self.assertAlmostEqual(current_xy[0,1],current_xy[4,1])
            reference_xy=np.array(ref['views']['lidar_top']['corners_pixels'])
            self.assertGreater(abs(reference_xy[0,1]-reference_xy[4,1]),1.)
            ppm=mapping['views']['lidar_top']['pixels_per_meter']
            moved=apply_action(state(current),dict(id='axis_adjustment',type='adjust',terminal=False,
                dx_px=5,dy_px=-1,dyaw_deg=2),ppm,mapping)
            folder=root/'round';folder.mkdir();(folder/'state.json').write_text(json.dumps(moved))
            (folder/'before_state.json').write_text(json.dumps(state(current)))
            new_map,new_ref=prepare_render_mapping(folder,folder/'state.json',source,root/'views',root/'reference','same-target')
            updated=render_views(source,folder/'state.json',None,folder/'views',reference_mapping=new_map)
            new_reference=json.loads((new_ref/'pixel_mapping.json').read_text())
            validate_same_target_reference(new_reference,updated)
            self.assertFalse(np.allclose(mapping['view_to_lidar'],updated['view_to_lidar']))
            np.testing.assert_allclose(recover_corners(new_reference),correct,atol=1e-12)
            vocab=dict(count=125,pixels_per_meter=ppm,actions=build_actions(ppm))
            payload=build_request(folder/'views',vocab,updated,reference_example=new_ref,reference_mode='same-target',stage='adjustment')
            self.assertIn('attached to the CURRENT box',payload['questions']['x']['instructions'])
            self.assertIn('box +X',payload['input'][-8]['text'])
            self.assertNotIn('points forward',payload['questions']['x']['instructions'])
