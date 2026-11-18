import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from model_tools.crop_mapping import expand_mapping, recover_corners
from model_tools.render_views import render_lidar


class CropMappingTests(unittest.TestCase):
    def test_expand_preserves_scales_and_reference_geometry(self):
        corners=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [1.,2.]])
        bounds=np.array([[-1.,-1.,.5],[1.,1.,2.5]])
        mapping=dict(bounds_lidar=bounds.tolist(),views={})
        for name,axes in [('lidar_top',[0,1]),('lidar_front',[0,2]),('lidar_side',[1,2])]:
            _,mapping['views'][name]=render_lidar(np.empty((0,3)),corners,bounds,axes,128)
        mapping['views']['camera_crop']=dict(crop_xyxy=[44,44,56,56],size_wh=[12,12])
        original=copy.deepcopy(mapping)
        np.testing.assert_allclose(recover_corners(mapping),corners)
        with tempfile.TemporaryDirectory() as directory:
            Image.new('RGB',(100,100)).save(Path(directory)/'camera.png')
            meta=dict(dataset='nuScenes',data_root=directory,camera_filename='camera.png',
                      lidar_to_global=np.eye(4).tolist(),global_to_camera=np.eye(4).tolist(),
                      camera_intrinsic=[[10,0,50],[0,10,50],[0,0,1]])
            unchanged,changed=expand_mapping(mapping,corners,meta)
            self.assertFalse(changed)
            self.assertEqual(unchanged,mapping)
            moved=corners+[2.,0.,0.]
            expanded,changed=expand_mapping(mapping,np.r_[corners,moved],meta)
            self.assertTrue(changed)
            self.assertEqual(mapping,original)
            self.assertTrue(np.all(moved<=expanded['bounds_lidar'][1]))
            for name in ['lidar_top','lidar_front','lidar_side']:
                old,new=mapping['views'][name],expanded['views'][name]
                self.assertEqual(old['pixels_per_meter'],new['pixels_per_meter'])
                self.assertGreaterEqual(new['size_wh'][0],old['size_wh'][0])
                self.assertGreaterEqual(new['size_wh'][1],old['size_wh'][1])
            self.assertGreater(expanded['views']['camera_crop']['crop_xyxy'][2],56)
            self.assertLessEqual(expanded['views']['camera_crop']['crop_xyxy'][2],100)

    def test_recovery_rejects_inconsistent_views(self):
        view=dict(lidar_axes=[0,1],pixels_per_meter=10.,u_min_m=0.,v_max_m=1.,corners_pixels=[[0,0]]*8)
        mapping=dict(views={name:copy.deepcopy(view) for name in ['lidar_top','lidar_front','lidar_side']})
        mapping['views']['lidar_front']['corners_pixels'][0]=[10,0]
        with self.assertRaises(ValueError):
            recover_corners(mapping)
