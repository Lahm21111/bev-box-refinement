# 模型执行工具

Gemini Decisions 先判断接受/修正/删除，选择修正时再分别判断X、Y、yaw七选一并合成一次变换；运行入口用以下工具执行。
这些是本地Python函数，当前 Decisions API 不通过自由生成的 function call 调用工具。

| 模块 | 函数 | 用途 |
|---|---|---|
| `model_tools.geometry` | `transform(points, matrix)` | 用4×4矩阵转换3D点 |
| `model_tools.geometry` | `project(world, sensor)` | 世界坐标点投影到相机，返回像素与深度 |
| `model_tools.geometry` | `perturb_box(corners, dx, dy, yaw_deg)` | 绕框中心转动，再在LiDAR水平面平移 |
| `model_tools.actions` | `build_actions(ppm)` | 保留旧动作文件兼容格式，校验像素尺度；不发送组合候选 |
| `model_tools.actions` | `validate_actions(vocabulary, mapping)` | 校验动作和BEV像素映射，返回ID到动作的字典 |
| `model_tools.actions` | `apply_action(state, action, scale)` | 返回更新后的框状态，不修改传入状态 |
| `model_tools.render_views` | `render_views(...)` | 保存当前框四张纯图及像素映射，返回映射字典 |

```python
from model_tools.actions import apply_action
from model_tools.render_views import render_views

updated = apply_action(state, chosen_action, bev_pixels_per_meter)
# 将 updated 写入 JSON，作为 offset 参数。
mapping = render_views(
    source="outputs/box_crops",
    offset="outputs/my_run/state.json",
    data_root="data/bench2drive",
    output="outputs/my_run/views",
    reference_mapping="outputs/small_offset_clean_views/pixel_mapping.json",
)
```

`apply_action` 的图像Y向下，转换为LiDAR Y时取反。
`accept_box` 保留几何并标记为 `accepted`；`delete_box` 标记为 `deleted`，
运行入口收到后结束循环。原始数据集不被修改。

渲染工具的 `reference_mapping` 固定原裁剪范围和像素尺度，以保证每轮的5/10像素
始终含义一致。更新后的框如果越出固定LiDAR范围，工具会报错，不会偷偷改变缩放。
图片只包含当前红框与点云或RGB，不画原框、坐标、图例、文字。

运行时若框超出当前裁剪范围，会自动扩大窗口，保持各LiDAR视图每米像素数不变。
相机保持原图像素单位，只扩展裁剪到原照片边界。同车正确参考会按同一新范围重绘。
扩图记录保存在对应轮次的 `crop_update.json`，新参考保存在 `reference/`。
这能避免固定窗口越界导致的中断，不改变模型选择的动作或框几何，也不能纠正模型漂移。
独立render_views工具仍检查给定固定范围；决策运行器负责先扩图再渲染。
