"""Orthographic crops aligned to the current box's own XYZ axes."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geometry import box_pose, transform, project
from .nuscenes_views import sample_identity, project_corners


PLANES = [('lidar_top',[0,1]), ('lidar_front',[1,2]), ('lidar_side',[0,2])]


def camera_path(meta, data_root=None):
    if meta.get('dataset') == 'nuScenes':
        root = Path(data_root) if data_root is not None else Path(meta['data_root'])
        return root / meta['camera_filename']
    root = Path(data_root) if data_root is not None else Path('data/bench2drive')
    channel = 'rgb_' + meta['camera'].removeprefix('CAM_').lower()
    return root / meta['route'] / 'camera' / channel / f'{meta["frame"]}.jpg'


def camera_pixels(corners, meta):
    if meta.get('dataset') == 'nuScenes':
        return project_corners(corners, meta)
    world = transform(corners, np.linalg.inv(meta['sensors']['LIDAR_TOP']['world2lidar']))
    pixels, depth = project(world, meta['sensors'][meta['camera']])
    if np.any(depth <= .1):
        raise ValueError('Box crosses camera near plane')
    return pixels


def make_box_mapping(corners, meta, reference=None, extra_corners=(), data_root=None,
                     padding=.2, max_size=1024):
    pose = box_pose(corners)
    inverse = np.linalg.inv(pose)
    union = np.concatenate([np.asarray(corners)] + [np.asarray(c) for c in extra_corners])
    local = transform(union, inverse)
    margin = np.maximum(np.ptp(local, axis=0)*padding, .15)
    bounds = np.array([local.min(0)-margin,local.max(0)+margin])
    views = {}
    for name, axes in PLANES:
        u,v = axes
        # Keep metric action units unchanged when changing the viewing orientation.
        old = reference['views'][name] if reference else None
        scale = old['pixels_per_meter'] if old else (max_size-1)/np.ptp(bounds[:,axes],axis=0).max()
        size = np.ceil((bounds[1]-bounds[0])[axes]*scale).astype(int)+1
        pixels = transform(corners, inverse)
        views[name] = dict(lidar_axes=axes, pixels_per_meter=float(scale),size_wh=size.tolist(),
            u_min_m=float(bounds[0,u]),v_max_m=float(bounds[1,v]),
            corners_pixels=np.c_[(pixels[:,u]-bounds[0,u])*scale,(bounds[1,v]-pixels[:,v])*scale].tolist(),
            axis_coordinate_system='box')
    pixels = camera_pixels(union, meta)
    with Image.open(camera_path(meta,data_root)) as image:
        size = np.array(image.size)
    pad = np.maximum(np.ptp(pixels,axis=0)*padding,12)
    rect = np.r_[np.floor(np.maximum(pixels.min(0)-pad,0)),np.ceil(np.minimum(pixels.max(0)+pad,size))].astype(int)
    if reference:
        old = np.asarray(reference['views']['camera_crop']['crop_xyxy'])
        rect = np.r_[np.minimum(rect[:2],old[:2]),np.maximum(rect[2:],old[2:])]
    current_pixels = camera_pixels(corners,meta)
    views['camera_crop'] = dict(size_wh=(rect[2:]-rect[:2]).tolist(),crop_xyxy=rect.tolist(),
        corners_pixels=(current_pixels-rect[:2]).tolist())
    bounds_corners = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    lidar_bounds = transform(bounds_corners,pose)
    identity = sample_identity(meta)
    return dict(dataset=meta.get('dataset','Bench2Drive'),frame=identity['frame'],box_id=identity['box_id'],
        view_frame='box',action_coordinate_system='box',view_to_lidar=pose.tolist(),lidar_to_view=inverse.tolist(),
        bounds_view=bounds.tolist(),bounds_lidar=[lidar_bounds.min(0).tolist(),lidar_bounds.max(0).tolist()],
        views=views,coordinates='Current box local XYZ: X length, Y width, Z height; all three views share this frame')


def render_box_views(source, offset, data_root, output, reference_mapping=None,
                     previous_offset=None, padding=.2,max_size=1024):
    from .render_views import render_lidar,draw_box
    from .crop_mapping import recover_corners
    source,output=Path(source),Path(output)
    meta=json.loads((source/'metadata.json').read_text())
    state=json.loads(Path(offset).read_text());identity=sample_identity(meta)
    if any(state[k]!=identity[k] for k in ['route','frame','box_id']):
        raise ValueError('Box state and source belong to different samples')
    corners=np.asarray(state['perturbed_corners_lidar'])
    previous=None
    if previous_offset:
        old=json.loads(Path(previous_offset).read_text())
        if any(old[k]!=identity[k] for k in ['route','frame','box_id']):
            raise ValueError('Previous box belongs to another sample')
        previous=np.asarray(old['perturbed_corners_lidar'])
    reference=json.loads(Path(reference_mapping).read_text()) if reference_mapping else None
    if reference and (reference['frame'],reference['box_id'])!=(identity['frame'],identity['box_id']):
        raise ValueError('Reference belongs to another sample')
    if reference and reference.get('view_frame')=='box':
        mapping=reference
    else:
        extras=([recover_corners(reference)] if reference else [])+([previous] if previous is not None else [])
        mapping=make_box_mapping(corners,meta,reference,extras,data_root,padding,max_size)
    if meta.get('dataset')=='nuScenes':
        root=Path(data_root) if data_root is not None else Path(meta['data_root'])
        points=np.fromfile(root/meta['lidar_filename'],dtype=np.float32).reshape(-1,5)[:,:3]
    else:
        import laspy
        root=Path(data_root) if data_root is not None else Path('data/bench2drive')
        points=transform(np.asarray(laspy.read(root/meta['route']/'lidar'/f'{meta["frame"]}.laz').xyz),meta['ego_to_lidar'])
    inverse=np.asarray(mapping['lidar_to_view']);bounds=np.asarray(mapping['bounds_view'])
    local=transform(points,inverse);local=local[np.all((local>=bounds[0])&(local<=bounds[1]),axis=1)]
    current=transform(corners,inverse)
    if np.any(current<bounds[0]-1e-8) or np.any(current>bounds[1]+1e-8):
        raise ValueError('Current box exceeds the shared box-frame crop')
    old_local=transform(previous,inverse) if previous is not None else None
    output.mkdir(parents=True,exist_ok=True);views={}
    for name,axes in PLANES:
        image,views[name]=render_lidar(local,current,bounds,axes,max_size,old_local,mapping['views'][name])
        views[name]['axis_coordinate_system']='box';image.save(output/f'{name}.png')
    pixels=camera_pixels(corners,meta)
    with Image.open(camera_path(meta,data_root)) as raw:camera=raw.convert('RGB')
    draw=ImageDraw.Draw(camera)
    if previous is not None:draw_box(draw,camera_pixels(previous,meta),'#ffd54f',3,dashed=True)
    draw_box(draw,pixels,'#ff604b',3)
    rect=np.array(mapping['views']['camera_crop']['crop_xyxy']);camera.crop(tuple(rect)).save(output/'camera_crop.png')
    views['camera_crop']=dict(crop_xyxy=rect.tolist(),size_wh=(rect[2:]-rect[:2]).tolist(),corners_pixels=(pixels-rect[:2]).tolist())
    result=dict(mapping,views=views,offset=state['offset'],seed=state.get('seed',0),
        image_content='current red box and point cloud / camera RGB; projections use the current box frame')
    if previous is not None:result.update(previous_corners_lidar=previous.tolist(),image_content='previous yellow dashed box and updated red box in the SAME current-box frame')
    (output/'pixel_mapping.json').write_text(json.dumps(result,indent=2))
    return result
