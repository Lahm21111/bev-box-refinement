"""Save exact model-input/output images separately from shared-frame overlays."""
import json
from pathlib import Path
import shutil

import matplotlib
from PIL import Image, ImageDraw, ImageFont


VIEWS = [('lidar_top','Box XY / BEV'), ('lidar_front','Front'),
         ('lidar_side','Side'), ('camera_crop','Camera')]


def save_actual_comparison(output, before, after):
    """Before is a byte-for-byte copy of the actual input, never reprojected."""
    output, before, after = map(Path, (output,before,after))
    output.mkdir(parents=True,exist_ok=True)
    for label,source in [('before',before),('after',after)]:
        target=output/label
        target.mkdir(exist_ok=True)
        for name,_ in VIEWS:
            shutil.copy2(source/f'{name}.png',target/f'{name}.png')
        shutil.copy2(source/'pixel_mapping.json',target/'pixel_mapping.json')
    before_mapping=json.loads((before/'pixel_mapping.json').read_text())
    after_mapping=json.loads((after/'pixel_mapping.json').read_text())
    box_mode=before_mapping.get('view_frame')=='box'
    fontpath=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    font=ImageFont.truetype(str(fontpath),24);small=ImageFont.truetype(str(fontpath),20)
    width,gap,header,label_height=700,16,100,34
    rows=[]
    for name,title in VIEWS:
        images=[]
        for side in ['before','after']:
            with Image.open(output/side/f'{name}.png') as raw:
                image=raw.convert('RGB')
                images.append(image.resize((width,round(image.height*width/image.width)),Image.Resampling.LANCZOS))
        row=Image.new('RGB',(2*width+gap,max(im.height for im in images)+label_height),'#101c2c')
        ImageDraw.Draw(row).text((12,4),title,font=small,fill='white')
        for x,image in zip([0,width+gap],images):
            row.paste(image,(x,label_height))
        rows.append(row)
    canvas=Image.new('RGB',(2*width+gap,header+sum(row.height for row in rows)+gap*(len(rows)-1)),'#101c2c')
    draw=ImageDraw.Draw(canvas)
    draw.text((12,8),'Before: actual input images',font=font,fill='white')
    draw.text((width+gap+12,8),'After: updated images',font=font,fill='white')
    note=('Each column uses its OWN box XYZ frame; both current boxes are axis-aligned.' if box_mode
          else 'Exact input and updated images; no reprojection of the before images.')
    draw.text((12,48),note,font=small,fill='#b5c9d8')
    draw.text((12,74),'View rotation / recentering can change the point-cloud appearance.',font=small,fill='#b5c9d8')
    y=header
    for row in rows:
        canvas.paste(row,(0,y));y+=row.height+gap
    canvas.save(output/'before_after.png')
    (output/'comparison_info.json').write_text(json.dumps(dict(
        before_source=str(before.resolve()),after_source=str(after.resolve()),
        before_is_exact_input_copy=True,before_frame=before_mapping.get('view_frame','lidar'),
        after_frame=after_mapping.get('view_frame','lidar'),
        note='before_after.png uses each state own frame. Top-level overlay PNGs use the post-action shared frame.'),indent=2))
    overlays=''.join(f'<a href="{name}.png">{title} overlay</a> ' for name,title in VIEWS)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Actual before and after</title>'
        '<style>body{background:#101c2c;color:white;font-family:sans-serif}img{max-width:100%}a{color:#9ad5ff}</style>'
        '<p>Before is the actual model input. After is the updated state in its own view frame.</p>'
        '<img src="before_after.png"><p>Separate shared-frame diagnostic overlays: '+overlays+'</p>')
    return output/'before_after.png'
