# 单框输入与独立轴修正

在项目根目录执行：

```bash
python -m model_runtime.box_refinement_pipeline \
  --box outputs/status_examples_10/case_03/state.json \
  --reference-example outputs/status_examples_10/case_05/views \
  --reference-mode same-target \
  --max-rounds 10 --output outputs/single_box_pipeline
```

`--box` 接受8×3的LiDAR XYZ角点数组、`{"corners_lidar": [...]}`，
或带 `perturbed_corners_lidar` 的已有状态JSON。角点顺序与数据集相同，
不能任意打乱。坐标为LiDAR传感器坐标，单位米。

框对应的点云、RGB、标定信息来自 `--source` 的元数据和 `--data-root` 的数据集。
默认使用Nissan样例的 `outputs/box_crops`。其他车辆必须指定相匹配的source，
例如Impala为 `outputs/examples/impala_00160_7477/source`。
带route/frame/box_id的输入会核对身份。

省略 `--reference-example` 和 `--reference-mode` 可不使用正确参考。
使用同车正确参考时，初始图片自动沿用参考的裁剪和像素映射；也可通过
`--reference-mapping` 显式指定。后续所有轮次固定裁剪和像素尺度。

流程为：

1. 从输入框生成四张纯净视图。
2. 第一次API调用只判断 `accept_box / refine_box / delete_box`。
3. 选到修正时，第二次API调用包含 `x`、`y`、`yaw` 三个独立问题，每个七选一。
4. 合并三分量，先绕当前框中心旋转，再平移；尺寸、Z、pitch、roll保持不变。
5. 生成更新后的四视图，重新进入状态判断，直至接受、删除、停滞或达到轮次上限。

| 问题 | negative | zero | positive |
|---|---:|---:|---:|
| x | BEV向左1/5/10像素 | 不动 | BEV向右1/5/10像素 |
| y | BEV向上1/5/10像素 | 不动 | BEV向下1/5/10像素 |
| yaw | BEV顺时针1/2/5度 | 不转 | BEV逆时针1/2/5度 |

默认使用全部档位。显式传入 `--pixel-step` 或 `--yaw-step` 时，会把对应分量限制为该单档的正/零/负三个选项。
三个问题共用同一组图，各自判断该分量，互相不以对方选择为条件；执行前不更新框。
候选按轴分别给出，不枚举343种组合。若三个分量均为零，则保留当前框并记为 `stalled`，
不把它自动当成接受，也不强制进行一个偏移。

输出目录包含：

- `pipeline.json`：本次输入与配置。
- `input/state.json`、`input/views/`：输入框与初始图片。
- `run/round_001/gate/`：状态阶段的请求、响应和决策。
- `run/round_001/adjustment/`：需要修正时的三个问题及响应。
- `run/round_001/decision.json`：三个分量的选择、各自置信度与合成动作。
- `run/round_001/views/`：修正后的图片。
- `run/final_state.json`、`run/history.json`：最终框和完整决策过程。

`offset` 表示相对于本次输入框累计执行的变化，不是相对于数据集真值的偏移。
达到轮次上限时，状态可能仍为 `adjusting`，不能解读为已接受。
可在命令末尾加 `--dry-run` 检查输入与两阶段请求而不联网。
所有发给模型的指令为英文，数值真值与预期修正答案不会发送。

已有四视图和状态时，仍可直接用 `python -m model_runtime.gemini_box_decision`
配合 `--images / --state / --actions / --source` 运行同一套决策循环。

每轮现在还会保存 `before_state.json` 和 `comparison/` 下的四张前后框叠加图：
黄色虚线是本轮修改前的框，红色实线是修改后的框。每步对比的都是紧邻的前后状态。
接受、删除、停滞时几何不变，两框重合；删除状态仍在图中显示框供复核。
图片不含文字、坐标或图例。模型下一轮继续读取只有当前红框的 `views/`。
打开运行目录的 `index.html` 可按轮查看全部对比图；pipeline入口对应 `run/index.html`。

设置 `--max-rounds 100 --visualize-every 5` 可最多运行100轮，每5轮保存前后框对比。
提前结束和最后一轮也会保存；每轮模型使用的纯净四视图继续更新。
浏览页仅收录实际保存的对比轮次，每张比较该轮修改前后的相邻状态。

## nuScenes输入

nuScenes导出目录可作为 `--source outputs/nuscenes_box_crops`，也可直接作为同车正确参考。
使用 `--box outputs/nuscenes_random_offset/offset.json` 输入随机扰动框。
数据路径默认从元数据读取；原生LiDAR坐标的X朝右、Y朝前，正视为XZ、侧视为YZ。
发送给模型的英文视图说明和重投影均按nuScenes约定自动切换，动作仍使用BEV原图像素。

运行时若框超出当前裁剪范围，会自动扩大窗口，保持各LiDAR视图每米像素数不变。
相机保持原图像素单位，只扩展裁剪到原照片边界。同车正确参考会按同一新范围重绘。
扩图记录保存在对应轮次的 `crop_update.json`，新参考保存在 `reference/`。
这能避免固定窗口越界导致的中断，不改变模型选择的动作或框几何，也不能纠正模型漂移。
独立render_views工具仍检查给定固定范围；决策运行器负责先扩图再渲染。

## 当前box局部坐标三视图

新的 `box_refinement_pipeline` 默认 `--view-frame box`：点云先转换到当前框的局部XYZ，
X沿长度、Y沿宽度、Z沿高度，再分别生成XY俯视、YZ正视、XZ侧视。
局部原点为框中心，因此当前框在三视图中轴对齐，点云相对框的偏移/朝向体现误差。
相机图仍使用真实相机透视，不旋转RGB图像。

每轮的正确参考、当前框和前后框对比都共用当前框的观察坐标系；
动作执行后，下一轮按新框的坐标系重新生成点云图和正确参考。
X/Y动作使用当前box局部BEV原图像素，正X向右、正Y像素向下（对应局部-Y）。
yaw绕当前box局部+Z旋转，正值在该BEV中逆时针。
框中心的传感器坐标变化由局部基向量转换得到；有倾斜时可以带来少量传感器Z变化，尺寸保持不变。
动作指的是框在当前坐标系中的移动，而重绘后点云在屏幕上可能向相反方向变化。

旧Gemini/Sol结果继续保留原来的LiDAR固定轴含义，不会自动改写为新的实验结果。
可用 `--view-frame lidar` 显式恢复旧观察/动作坐标。

```bash
python -m model_runtime.box_refinement_pipeline \
  --box outputs/bench2drive_sedan_restart_001/input/state.json \
  --source outputs/bench2drive_sedan_restart_001/input/source \
  --reference-example outputs/bench2drive_sedan_restart_001/input/reference \
  --reference-mode same-target --view-frame box \
  --max-rounds 100 --visualize-every 5 \
  --output outputs/sedan_box_local
```

### 真实Before/After与多轮Decision调用

`comparison/before/` 逐字节保存本轮实际输入图片；`comparison/after/` 保存更新后的图片。
`comparison/before_after.png` 的两列分别使用各自box的坐标系，所以当前框均轴对齐。
点云可能因坐标系旋转和重新居中而改变屏幕位置，不应把两个像素坐标系直接相减。
`comparison/` 顶层四张PNG则是共享修改后坐标系的黄/红框叠加诊断图，二者不混称。
运行目录的 `index.html` 默认展示真实输入/输出对比，并链接诊断叠加图。

代码固定调用 `gemini-decision-1` 的 Decisions REST API，不是普通生成模型。
每轮状态判断调用一次API；需要修正时第二次调用包含X、Y、yaw三个独立的7选1问题。
`--max-rounds 100` 设置轮次上限，接受、删除、全零停滞会提前结束。
收到HTTP 429时按Retry-After或错误正文的等待时间重试同一请求，最多额外重试3次。
不使用 `--dry-run` 或 `--response-file` 时才会调用真实API。
