"""Render the selected CAM_FRONT frame using official nuScenes devkit boxes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from model_runtime.gemini_box_decision import create_run_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=Path('outputs/nuscenes_box_crops/metadata.json'),
                        help='Use the exact same dataset, frame and vehicle as this saved export')
    parser.add_argument('--output', type=Path, default=Path('outputs/nuscenes_devkit_camera'))
    parser.add_argument('--all-boxes', action='store_true', help='Render all visible annotations in this camera')
    args = parser.parse_args()
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    except ImportError as error:
        parser.error(f'nuScenes devkit dependency is unavailable ({error}). '
                     'Use the existing drivetransformer environment, or install nuscenes-devkit.')
    metadata = json.loads(args.metadata.read_text())
    nusc = NuScenes(version=metadata['version'], dataroot=metadata['data_root'], verbose=False)
    sample = nusc.get('sample', metadata['sample_token'])
    annotation = nusc.get('sample_annotation', metadata['box_token'])
    if annotation['sample_token'] != sample['token']:
        raise ValueError('Selected box does not belong to this frame')
    camera_token = sample['data']['CAM_FRONT']
    image_path, boxes, intrinsic = nusc.get_sample_data(
        camera_token, box_vis_level=BoxVisibility.ANY,
        selected_anntokens=None if args.all_boxes else [metadata['box_token']])
    if not boxes:
        raise ValueError('No selected annotations are visible in CAM_FRONT')
    with Image.open(image_path) as raw:
        image = raw.convert('RGB')
    width, height = image.size
    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(image)
    for box in boxes:
        # Both sensor-frame conversion and box projection/rendering are official
        # SDK operations. No project-local quaternion or camera projection is used.
        box.render(ax, view=intrinsic, normalize=True, colors=('#ff604b',)*3, linewidth=2)
    ax.set_xlim(-.5, width-.5)
    ax.set_ylim(height-.5, -.5)
    ax.set_axis_off()
    output = create_run_directory(args.output)
    fig.savefig(output / 'camera_full.png', dpi=dpi, pad_inches=0)
    plt.close(fig)
    with Image.open(output / 'camera_full.png') as full:
        if full.size != (width, height):
            raise ValueError('Rendered image size differs from original image')
        # Reuse the exact previous crop so the two renderers are easy to compare.
        mapping_path = args.metadata.parent / 'pixel_mapping.json'
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text())
            full.crop(tuple(mapping['views']['camera_crop']['crop_xyxy'])).save(output / 'camera_crop.png')
    projected = {box.token: view_points(box.corners(), intrinsic, normalize=True)[:2].T.tolist()
                 for box in boxes}
    write_json(output / 'metadata.json', dict(renderer='nuScenes.get_sample_data + Box.render',
        version=metadata['version'], data_root=metadata['data_root'], sample_token=sample['token'],
        camera_token=camera_token, target_box_token=metadata['box_token'], all_boxes=args.all_boxes,
        box_count=len(boxes), image_path=str(image_path), size_wh=[width,height],
        corners_pixels=projected,
        note='Official Box.render includes a short line indicating the front of the box.'))
    print(f'CAM_FRONT: {camera_token}; {len(boxes)} box(es); {width}x{height}')
    print(f'Saved to {output.resolve()}')


if __name__ == '__main__':
    main()
