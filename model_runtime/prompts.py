"""View descriptions and decision instructions sent to Gemini."""

VIEWS = [
    ('lidar_top', 'Image 1: LiDAR bird\'s-eye view (BEV), XY plane. Image right is LiDAR +X (forward); image up is LiDAR +Y (vehicle right).'),
    ('lidar_front', 'Image 2: LiDAR front view, YZ plane. Image right is LiDAR +Y (vehicle right); image up is +Z (height).'),
    ('lidar_side', 'Image 3: LiDAR side view, XZ plane. Image right is LiDAR +X (forward); image up is +Z (height).'),
    ('camera_crop', 'Image 4: Front-camera RGB crop of the same target. Red lines are the perspective projection of the current 3D box.'),
]


REFERENCE_INSTRUCTIONS = '''CORRECT REFERENCE EXAMPLE - A DIFFERENT VEHICLE, NOT THE TARGET TO EDIT.
The following four images show an unperturbed dataset annotation and illustrate
what a satisfactory 3D box looks like. In BEV, the box position and heading match
the vehicle footprint; in front and side views, the visible vehicle surfaces
are consistent with the box; the camera projection encloses the same vehicle.
The box represents the full vehicle even when LiDAR returns cover only its
visible surfaces. Empty space on occluded sides is normal. Ground and neighboring
objects need not be enclosed. Use this example as the visual alignment standard.
Use it only as an example of alignment quality. Do not copy its pixel locations,
box dimensions, yaw, or meter-to-pixel scale to the current target. The example
and target are different vehicles with different crops and scales. Do not
choose an action for the reference. Choose an action only for CURRENT TARGET.
'''


INSTRUCTIONS = '''Refine the CURRENT TARGET vehicle's 3D bounding box using its four synchronized views.
If a CORRECT REFERENCE EXAMPLE is supplied, it is a different vehicle illustrating
acceptable alignment. Evaluate and act on CURRENT TARGET only. Image 1 through
Image 4 below always refer to the four images labeled CURRENT TARGET, not the reference.
The red wireframe is the current box. Blue/green points are LiDAR returns colored
only by height, not semantic segmentation labels. The first three images are
orthographic projections of the same local point cloud; the fourth is a camera
perspective view. Only horizontal LiDAR-plane translation and yaw adjustment are
allowed. Keep box length, width, height, Z, pitch, and roll unchanged.

Action coordinate conventions:
1. All dx_px and dy_px values are pixel displacements in the ORIGINAL PNG of
   Image 1 (BEV), not camera pixels or pixels in Images 2 and 3.
   Positive dx_px moves right in Image 1; negative dx_px moves left.
   Positive dy_px moves down in Image 1; negative dy_px moves up.
2. LiDAR X points forward, Y points toward vehicle right, and Z points up.
   Convert BEV pixel translation to meters as follows:
   delta_X_m = dx_px / BEV_pixels_per_meter
   delta_Y_m = -dy_px / BEV_pixels_per_meter
3. dyaw_deg is the rotation increment around LiDAR +Z through the CURRENT box
   center. Positive yaw rotates LiDAR +X toward +Y, appearing counterclockwise
   in Image 1; negative yaw appears clockwise. Rotate about the current center
   first, then translate.
4. One 3D action updates all three LiDAR projections and the camera projection.
   Do not apply the same pixel displacement directly to the other views: each
   has its own projection and pixel scale.
5. All increments are relative to the CURRENT box, not cumulative offsets
   measured from the initial box.

The first-stage decision has already selected refine_box.
Answer three independent questions, one for BEV X, one for BEV Y, and one for yaw.
Each question offers signed increments and zero. X/Y increments are -10, -5,
-1, 0, +1, +5, +10 original BEV pixels; yaw increments are -5, -2, -1, 0,
+1, +2, +5 degrees by default. Use exactly the choices supplied in criteria.
All questions refer to the SAME current box shown in these images. Do not assume
that a choice in one question has already been applied when answering another.
Select zero for a component that does not need correction. Do not invent a
movement just because another component needs correction. The controller will
combine the three components into one 3D update after all answers are available.

Choose the action expected to best improve 3D alignment using BEV position and
heading, front/side silhouettes, and camera-projection consistency. LiDAR samples
visible surfaces only; the far side of a vehicle may have no returns. Ground and
neighboring objects may also appear in the crop. Do not try to fit every point
or simply center the box on the centroid of visible returns.
Judge only the current four images and their view definitions. Do not assume
that the injected perturbation direction or a reference answer is known.'''


SAME_TARGET_REFERENCE_INSTRUCTIONS = '''CORRECT REFERENCE - THE SAME VEHICLE IN THE SAME FRAME.
These four images show the correct, unperturbed box for the exact vehicle that
appears in CURRENT TARGET. Both groups use identical point clouds/RGB, crop
boundaries, pixel scales, and view orientations. Only the red box differs.
Treat the reference red box as the desired final geometry. Compare corresponding
views to estimate how CURRENT TARGET should move and rotate toward it.
Choose actions only for CURRENT TARGET; never modify the correct reference.
Use the images to estimate the correction; no numeric correction is supplied.
This is an experiment with a known correct visual reference, not blind fitting.
'''

SAME_TARGET_INSTRUCTIONS = INSTRUCTIONS.replace(
    'If a CORRECT REFERENCE EXAMPLE is supplied, it is a different vehicle illustrating\n'
    'acceptable alignment. Evaluate and act on CURRENT TARGET only. Image 1 through\n'
    'Image 4 below always refer to the four images labeled CURRENT TARGET, not the reference.',
    'A CORRECT REFERENCE shows the SAME vehicle and frame with a correctly aligned box.\n'
    'The reference and CURRENT TARGET share identical backgrounds, crops, and pixel scales.\n'
    'Compare the red boxes in corresponding views and move CURRENT TARGET toward the\n'
    'reference geometry using the independent component choices.\n'
    'Do not modify the reference. Action pixel coordinates refer to CURRENT TARGET BEV.'
).replace(
    'Judge only the current four images and their view definitions. Do not assume\n'
    'that the injected perturbation direction or a reference answer is known.',
    'Judge the visual difference between CORRECT REFERENCE and CURRENT TARGET.\n'
    'Estimate correction from image evidence; no numeric injected offset is provided.'
)


GATE_INSTRUCTIONS = """Evaluate the CURRENT TARGET vehicle box using all four synchronized views.
Choose exactly one of the three criteria: accept_box, refine_box, or delete_box.
This is a quality decision before selecting any individual movement. Assess
whether ANY allowed BEV XY translation or yaw correction is needed, considering
all plausible correction directions together rather than one specific action.
Accept if alignment is already satisfactory at the available action resolution
(5 or 10 original BEV pixels, 1 or 5 degrees), with no clear supported improvement.
Refine if the target is valid and a horizontal translation or yaw correction
would clearly improve alignment. Do not reject a valid target for position errors.
Delete only a false or invalid annotation with no corresponding valid vehicle.
Sparse returns, occlusion, and boundary clipping alone do not justify deletion.
The red wireframe is the current 3D box. LiDAR colors encode height only.
Image 1 is BEV XY: right is LiDAR +X, up is +Y. Image 2 is front YZ:
right is +Y, up is +Z. Image 3 is side XZ: right is +X, up is +Z.
Image 4 is the front-camera perspective projection of the same box.
LiDAR samples visible surfaces, not the full vehicle volume. Empty occluded
sides are normal; ignore ground and neighboring objects when judging alignment.
Only BEV XY and yaw can change; box dimensions and height remain fixed.
If a different-vehicle correct example is supplied, use it as an alignment
standard without copying its geometry. If a SAME TARGET correct reference is
supplied, compare corresponding red boxes against the desired reference geometry.
Decide only for CURRENT TARGET. No numerical ground-truth correction is supplied.
"""

VEHICLE_ENCLOSURE_INSTRUCTIONS = '''Vehicle enclosure objective:
Identify the points that plausibly belong to the CURRENT TARGET vehicle in the
LiDAR views and the corresponding vehicle in the camera image. Position and
orient the 3D box so that it tightly encloses those likely vehicle points, and
its camera projection tightly encloses the same visible vehicle. Both modalities must
support the same target; do not switch to a neighboring vehicle.
In BEV, align with the likely vehicle footprint and longitudinal direction.
In front and side views, contain the likely vehicle surfaces at their observed
positions. In the camera, check that the projected box covers the visible target
vehicle rather than an unrelated patch of road or background.
Tight enclosure is required, not mere overlap or loose containment. Within the
available fixed box dimensions, minimize avoidable empty margins around the
vehicle footprint, observed LiDAR surfaces, and camera target. Bring the box
boundaries close to supported vehicle boundaries while retaining the target's
visible body and likely vehicle points. Do not cut off one side of the vehicle
to make another side look tighter. Check tightness jointly across all views;
a correction that improves one view but worsens the overall fit is undesirable.
Exclude ground, poles, walls, and neighboring objects when identifying vehicle
points. Do not enlarge or reposition the box merely to include all points in
the crop. LiDAR returns sample visible surfaces only, so empty space on hidden
sides is normal. Judge camera tightness by the projected outer envelope and
plausible 3D geometry; hidden cuboid edges can cross the vehicle's image and do
not need to follow visible contours. Keep box dimensions fixed. Do not invent
size changes or use translation/yaw to compensate for an unchangeable size error.
At the status stage, request refinement when correction would improve this joint
enclosure and alignment; accept the current box when it already achieves
it at the available action resolution. At the adjustment stage, answer only the
supplied component question and prefer movements that improve this same objective.
'''


GATE_CRITERIA = {
    'accept_box': 'The current valid box tightly encloses the likely target-vehicle LiDAR points and the same camera vehicle, with no avoidable loose margins that the allowed XY/yaw actions can improve. Keep its geometry and stop.',
    'refine_box': 'The target is valid, but BEV XY translation or yaw correction would make its enclosure tighter and better aligned jointly with the likely vehicle points and the same camera vehicle. Continue to adjustment selection.',
    'delete_box': 'There is no corresponding valid vehicle target. Mark this false or invalid annotation deleted and stop.',
}


def axis_questions(pixel_step=None, yaw_step=None):
    """Independent per-axis questions over a shared image context."""
    from model_tools.actions import axis_options
    result = {}
    for axis, options in axis_options(pixel_step, yaw_step).items():
        unit = 'degrees' if axis == 'yaw' else 'original BEV pixels'
        negative, positive = {'x': ('left','right'), 'y': ('up','down'),
                              'yaw': ('clockwise in BEV','counterclockwise in BEV')}[axis]
        criteria = {}
        for key, option in options.items():
            value = option['value']
            criteria[key] = (f'Keep the current {axis} component unchanged.' if value == 0 else
                f'Change only {axis} by {value:+g} {unit}: {negative if value < 0 else positive}.')
        result[axis] = dict(type='choice', criteria=criteria)
    return result


def view_descriptions(mapping):
    if mapping.get('view_frame') == 'box':
        return [
            ('lidar_top', 'Image 1: Orthographic view along box Z onto local box XY. Image right is box +X (length); image up is box +Y (width).'),
            ('lidar_front', 'Image 2: Orthographic view along box X onto local box YZ. Image right is box +Y (width); image up is box +Z (height).'),
            ('lidar_side', 'Image 3: Orthographic view along box Y onto local box XZ. Image right is box +X (length); image up is box +Z (height).'),
            VIEWS[3],
        ]
    if mapping.get('dataset') != 'nuScenes':
        return VIEWS
    return [
        ('lidar_top', "Image 1: nuScenes LiDAR bird's-eye view (BEV), XY plane. Image right is LiDAR +X (right); image up is LiDAR +Y (forward)."),
        ('lidar_front', 'Image 2: nuScenes LiDAR front view, XZ plane. Image right is LiDAR +X (right); image up is +Z (height).'),
        ('lidar_side', 'Image 3: nuScenes LiDAR side view, YZ plane. Image right is LiDAR +Y (forward); image up is +Z (height).'),
        VIEWS[3],
    ]


BOX_FRAME_VIEW_RULES = """The view frame is attached to the CURRENT box, not the sensor or world.
All three point-cloud views share this current-box frame. The current box
therefore appears axis-aligned even when it has a yaw error; infer errors
from the relative orientation and position of the likely vehicle points.
Reference and current images in this round use the SAME current-box frame.
A positive dx moves the box along current box +X; positive dy moves it along box -Y.
Positive yaw turns box +X toward box +Y, counterclockwise in Image 1.
The controller converts these view-plane increments into sensor coordinates.
Actions describe how the BOX moves in the frame displayed NOW, not how the
point cloud should move on screen. After executing an action, views are
regenerated in the UPDATED box frame. The current box remains axis-aligned,
while the point cloud changes relative to it; recentering can make point
motion on screen appear opposite to the box translation action.
"""


def dataset_instructions(instructions, mapping):
    if mapping.get('view_frame') == 'box':
        instructions = instructions.replace('LiDAR', 'box-local').replace('LiDAR-plane','box-local-plane')
        instructions = instructions.replace('box-local X points forward, Y points toward vehicle right, and Z points up.',
            'Local box X follows box length, Y follows box width, and Z follows box height.')
        instructions = instructions.replace('Keep box length, width, height, Z, pitch, and roll unchanged.',
            'Keep box dimensions fixed. Translate along the displayed box X/Y axes and rotate around its Z axis.')
        instructions = instructions.replace('Only BEV XY and yaw can change; box dimensions and height remain fixed.',
            'Only local box XY translation and rotation about box Z are allowed; dimensions remain fixed.')
        return instructions + '\n\n' + BOX_FRAME_VIEW_RULES
    if mapping.get('dataset') != 'nuScenes':
        return instructions
    return instructions.replace(
        'LiDAR X points forward, Y points toward vehicle right, and Z points up.',
        'nuScenes LiDAR X points right, Y points forward, and Z points up.'
    ).replace(
        'Image 2 is front YZ:\nright is +Y, up is +Z. Image 3 is side XZ: right is +X, up is +Z.',
        'Image 2 is front XZ:\nright is +X, up is +Z. Image 3 is side YZ: right is +Y, up is +Z.'
    )
