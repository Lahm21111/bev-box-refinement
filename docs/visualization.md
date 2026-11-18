# Bench2Drive 单框多视图裁剪

默认示例使用 `data/bench2drive` 下的路线
`AccidentTwoWays_Town13_Route30356_Weather10_02-10-03-31-29`，
第 `00100` 帧、车辆 ID `34598`（Nissan Patrol），相机 `CAM_FRONT`。
使用数据集真值标注框；尚未接入检测器或偏移规划模型。

运行：

```bash
.venv/bin/python -m data_pipeline.visualize_box_crops
```

新环境安装依赖后可使用任意 Python 环境运行：

```bash
python -m pip install -r requirements.txt
python -m data_pipeline.visualize_box_crops --frame 100 --box-id 34598 --padding 0.2
```

可用 `--data-root`、`--route`、`--camera` 和 `--output` 指定输入输出。
框必须在指定相机中可见，且所有角点位于相机近裁面前方。

输出目录为 `outputs/box_crops/`：

- `overview.png`：俯视 XY、正视 YZ、侧视 XZ、相机裁剪拼图。
- `lidar_top.png`、`lidar_front.png`、`lidar_side.png`：独立点云裁剪图。
- `camera_crop.png`：带投影三维框的相机裁剪图。
- `camera_context.jpg`：完整相机图，红色为三维框，黄色为裁剪范围。
- `geometry.npz`：局部点云、世界和 LiDAR 框角点、相机投影及裁剪边界。
- `metadata.json`：样本标识、原始标注、标定、坐标变换与裁剪参数。

LiDAR 三视图共享同一份三维局部点云。先取框在 LiDAR 坐标系中的轴对齐范围，
每侧扩大对应尺寸的 20%（最小 0.15 米），保留范围内的所有点，包括地面及邻近物体。
三维框用红线绘制，点颜色随高度变化。坐标单位为米，X 向前、Y 向右、Z 向上。

Bench2Drive 的 `tools/data_collect.py` 在保存 LAZ 前将点转换到自车坐标系。
脚本使用 `world2lidar @ inverse(world2ego)` 将点云转回 LiDAR 传感器坐标系，
用标注中的 `world_cord` 和 `world2lidar` 转换框角点。
相机投影使用 `world2cam`，再将 CARLA 的 `(X,Y,Z)` 变换成针孔模型的 `(Y,-Z,X)`，
最后应用相机内参。

相机裁剪按投影框的二维外包矩形向各侧扩展 20%，并限制在原图范围内。
当前示例下侧边距受原图底边限制。图像不做拉伸；LiDAR 图使用等比例坐标轴。

## BEV 随机 XY / yaw 偏移

```bash
.venv/bin/python -m data_pipeline.random_bev_offset
```

每次运行随机生成 seed 并写入 `outputs/bev_random_offset/offset.json`。
X、Y 分别在 [-1, 1] 米内独立均匀采样，yaw 在 [-15, 15] 度内均匀采样。
围绕框中心绕 LiDAR Z 轴旋转后，在 LiDAR XY 轴平移，保留框尺寸和高度。
正 yaw 从 +X 转向 +Y。可用 `--max-xy`、`--max-yaw-deg` 调整范围，
用 `--seed 42` 复现同一组偏移。

`bev_comparison.png` 中黄色虚线为原框，红色实线为偏移框，
框内箭头表示车头方向，白色箭头表示中心位移。点云保持原位，
裁剪区域覆盖两个框并保留边距。`offset.json` 记录偏移及恢复原框所需的修正量，
`geometry.npz` 保存原框、偏移框和局部点云，原始数据集和此前输出不修改。

## 供视觉模型使用的纯图像

```bash
.venv/bin/python -m model_tools.render_views
```

读取已保存的随机偏移，输出到 `outputs/offset_clean_views/`：
`lidar_top.png`、`lidar_front.png`、`lidar_side.png`、`camera_crop.png`。
图片只绘制偏移后的红色框与点云或相机 RGB，无原框、图例、标题、坐标轴、
刻度、网格、方向箭头或画布留白。点云图保持等比例，长边默认 1024 像素。
裁剪范围覆盖原目标和偏移框并扩展 20%，避免偏移后丢失用于判断的目标点云。
相机裁剪限制在原图边界内。

`pixel_mapping.json` 单独保存每张图的像素映射、框角点和偏移参数。
LiDAR 图片中像素横坐标向右增大、纵坐标向下增大。
可根据各图的 `lidar_axes`、`pixels_per_meter`、`u_min_m`、`v_max_m`
将模型输出的像素坐标还原到 LiDAR 米制坐标；这些信息不画进图片。

## 小偏移与固定像素动作

```bash
.venv/bin/python -m data_pipeline.create_small_offset
.venv/bin/python -m model_tools.render_views --offset outputs/bev_small_offset/offset.json --output outputs/small_offset_clean_views --reference-mapping outputs/offset_clean_views/pixel_mapping.json
```

动作词表保存到 `outputs/bev_small_offset/actions.json`：
BEV 图像 X、Y 分别取 `[-10, -5, 0, 5, 10]` 像素，yaw 取
`[-5, -1, 0, 1, 5]` 度，共 124 种非零偏移组合（排除全零组合），
另加 `delete_box`（当前框应删除，删除并结束调整），总计 125 个候选。
偏移动作的 `type` 为 `adjust`，`terminal` 为 `false`；
删除动作的 `type` 为 `delete`，`terminal` 为 `true`，不包含平移或角度字段。
全零偏移动作已移除，单个维度仍可为零；非删除动作持续执行到最大轮次。
图像 X 向右，图像 Y 向下；LiDAR Y 在 BEV 中向上，因此需要反转平移符号。
yaw 正方向为 LiDAR +X 转向 +Y。动作先绕当前框中心旋转，再平移。

小偏移从原始框重新生成，X、Y 各随机选择 ±5/±10 像素，yaw 随机 ±1 度。
反向修正属于动作词表，几何上可以一步恢复；模型是否选对需要实际评估。
用 `--seed` 可复现。生成图片时固定前一版的三维裁剪边界和相机裁剪区域，
保持点云像素位置、每米像素数及图片尺寸一致，避免重新裁剪改变步长含义。
真值偏移及修正量单独保存在 `offset.json`，不要作为模型输入。

也可指定更大的初始偏移，动作候选仍保持原来的5/10像素、1/5度：

```bash
python -m data_pipeline.create_small_offset --dx-px 30 --dy-px 20 --yaw-deg 5 --output outputs/bev_medium_offset
python -m model_tools.render_views --offset outputs/bev_medium_offset/offset.json --output outputs/medium_offset_clean_views --reference-mapping outputs/offset_clean_views/pixel_mapping.json
python -m model_runtime.gemini_box_decision --images outputs/medium_offset_clean_views --state outputs/bev_medium_offset/offset.json --actions outputs/bev_medium_offset/actions.json --max-rounds 5 --output outputs/gemini_medium_run
```

## Gemini Decision 接入

运行与提示词说明见 [gemini-decision.md](gemini-decision.md)。
