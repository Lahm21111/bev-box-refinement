# Gemini Decision 单框修正

`model_runtime.gemini_box_decision` 通过官方 Decisions REST API 调用 `gemini-decision-1`，
每轮先用四张当前图片判断 `accept_box / refine_box / delete_box`。
只有选择 `refine_box` 才发起第二次请求，分别为X、Y、yaw回答三个七选一问题。
两阶段使用相同图片和参考案例；不在同一个请求中同时询问，确保修正阶段以第一阶段结果为前提。
发送给模型的指令、四视图说明、输入文字和动作候选描述统一使用英文，
提示词见 `model_runtime/prompts.py`。
使用 Python 标准库发送请求，无需安装 Google SDK；图像和几何操作使用现有可视化依赖。

```bash
# 在终端设置密钥，不要写入代码或提交到仓库。
export GEMINI_API_KEY='你的 API key'

# 单轮：请求、执行一次动作、保存新框及无标注元素的四视图。
.venv/bin/python -m model_runtime.gemini_box_decision

# 最多5轮；每轮固定裁剪和像素尺度，使用更新后的四张图重新决策。
.venv/bin/python -m model_runtime.gemini_box_decision --max-rounds 5 --output outputs/gemini_run_5

# 不联网、不需要密钥：检查真实请求，包括两个阶段的提示词、3个质量选项、每轴7个修正选项及图片。
.venv/bin/python -m model_runtime.gemini_box_decision --dry-run --output outputs/gemini_request_preview
```

支持 `GOOGLE_API_KEY` 作为备用环境变量。输出目录若已存在，会自动追加
`_001`、`_002` 等编号，保留之前的请求、响应及运行记录；可直接重复同一条命令。
默认使用 `outputs/small_offset_clean_views` 和 `outputs/bev_small_offset/actions.json`。
`--images`、`--state`、`--actions` 可指定其他样本或接续某轮输出，三者必须对应。
接续时设置 `--images 上轮目录/views --state 上轮目录/state.json --output 新目录`。
脚本会核对三张点云图中的框角点、像素尺度与当前状态一致。

## 视图与动作含义

可提供另一辆车的正确框作为参考，让模型看到合格的贴合案例：

```bash
python -m model_runtime.gemini_box_decision \
  --images outputs/medium_offset_clean_views \
  --state outputs/bev_medium_offset/offset.json \
  --actions outputs/bev_medium_offset/actions.json \
  --reference-example examples/correct_impala \
  --max-rounds 5 --output outputs/gemini_with_reference
```

`examples/correct_impala` 使用 Chevrolet Impala 的未施加扰动的原始标注框。
请求先给出4张正确示例图，再给出4张当前目标图，共8张。
全英文指令明确示例用于学习贴合标准，不能照搬其位置、尺寸、朝向或像素比例，
并且只允许对当前目标选择动作。当前目标的真值与恢复答案仍不会发送。
不传 `--reference-example` 时仍为原来的4图请求，方便对比效果。

同一车辆、同一帧的正确框和偏移框也可以同时输入，作为有正确答案参考的对照实验：

```bash
python -m model_runtime.gemini_box_decision \
  --images outputs/medium_offset_clean_views \
  --state outputs/bev_medium_offset/offset.json \
  --actions outputs/bev_medium_offset/actions.json \
  --reference-example examples/correct_nissan_00100 \
  --reference-mode same-target \
  --max-rounds 1 --output outputs/gemini_same_target_reference
```

该模式输入同一辆Nissan、第00100帧的正确4图和当前偏移4图，背景、裁剪及像素比例一致。
提示词明确要求比较两组红框，只调整当前框。不同于异车参考，此实验有意提供目标正确框的图像。
程序校验帧、车辆ID、三维裁剪、各视图尺寸和像素映射，防止把不同缩放误认为位置偏差。
每轮当前图会更新，正确参考不变。数字形式的真值偏移和修正量不会发送。

| 输入 | 投影 | 图像横向正方向 | 图像纵向向上 |
|---|---|---|---|
| lidar_top.png | BEV 俯视 XY | LiDAR +X（前方） | LiDAR +Y（右侧） |
| lidar_front.png | 正视 YZ | LiDAR +Y（右侧） | LiDAR +Z（高度） |
| lidar_side.png | 侧视 XZ | LiDAR +X（前方） | LiDAR +Z（高度） |
| camera_crop.png | 前相机透视 | 相机图像向右 | 相机图像向上 |

蓝绿点按高度着色，不是语义分割标签。红框表示当前框。
所有 `dx_px` / `dy_px` 都针对 **BEV 原始PNG像素**：X向右、Y向下。
图像尺寸或模型内部编码分辨率不改变候选动作的像素单位。
以当前 `111.61624321819004 px/m` 为例，5像素约0.0448米，10像素约0.0896米。

```text
LiDAR ΔX = BEV dx_px / pixels_per_meter
LiDAR ΔY = -BEV dy_px / pixels_per_meter
```

yaw 正方向为 LiDAR +X 转向 +Y，即当前BEV显示中的逆时针。
执行时先绕当前框中心转动，再做平移。框尺寸和高度保持不变。
其他三个视图根据更新后的三维框重新投影，不直接使用BEV像素位移。

修正阶段共3个独立问题，均针对同一组当前图：

- X：`-10 / -5 / -1 / 0 / +1 / +5 / +10` 原始BEV像素，正值向右。
- Y：`-10 / -5 / -1 / 0 / +1 / +5 / +10` 原始BEV像素，正值向下。
- yaw：`-5 / -2 / -1 / 0 / +1 / +2 / +5` 度，正值为BEV逆时针。

使用 `--pixel-step 10` 可将X/Y改为 `-10 / 0 / +10`，
使用 `--yaw-step 5` 可将yaw改为 `-5 / 0 / +5`。
每个问题分别归一化概率，不把XYZ/yaw组合枚举成候选。
收到全部分量后合并执行一次变换，并重画四视图进入下一轮。
第一阶段 `accept_box` 保留当前几何并结束，`delete_box` 标记删除并结束，
`refine_box` 进入修正阶段。最多执行 `--max-rounds` 轮，每轮调用1或2次API。
三个分量全零时状态记为 `stalled` 并结束，不强制移动或自动接受。
旧动作文件仍保留兼容的125项格式，运行时仅用于校验输入尺度；不向模型发送旧的组合词表。
终止决策保存在本地状态中，不修改数据集原始标注。

## 输出与验证

每轮 `round_001/` 等目录包含：

- `gate/`：第一阶段的 `request.json`、`instructions.txt`、`response.json`、`decision.json`。
- `adjustment/`：仅选择修正时保存第二阶段的同类文件；dry-run 同时预览两个阶段。
- `decision.json`：最终执行动作；`gate_decision` 保存第一阶段结果，`component_decisions` 保存各轴置信度与候选概率。
- 两阶段概率分别归一化，不直接比较不同阶段的置信度。
- `state.json`：应用动作后的本地框状态。
- `views/`：偏移动作执行后的四张纯图及像素映射；终止动作不重画。

顶层 `history.json` 保存历次决策，`final_state.json` 保存最终状态，
`run.json` 区分真实API调用、离线重放和请求预览。
默认分辨率使用 `ultra_high`，可通过 `--resolution` 修改。
模型输入仅包含四张当前图、视图定义、像素尺度和动作候选。
真值原框、预设扰动、恢复答案和随机种子不会发送给模型。
Decision API 返回候选概率，不生成自由文本解释。

```bash
.venv/bin/python -m unittest tests.test_gemini_box_decision -v

# 将两阶段原始响应整理成 {"gate": 第一阶段响应, "adjustment": 第二阶段响应} 的JSON文件。
# accept/delete 只需 gate；refine 的 adjustment.answers 必须包含 x、y、yaw。
# 默认choice为zero或negative_N/positive_N（N为对应步长）。离线重放一轮：
.venv/bin/python -m model_runtime.gemini_box_decision --response-file path/to/responses.json --output outputs/replay
```

单元检查覆盖图像请求、禁止真值信息进入请求、像素方向、三维刚体变换、
逆向恢复、接受/删除终止、两阶段分支、独立轴问题与调用次数、响应和概率校验。离线重放可验证代码链路，不能证明模型决策质量。

官方参考：
[Decisions API](https://aistudio.google.com/docs/decisions-api)、
[图片输入](https://aistudio.google.com/docs/decisions-api#media)。

## 十例状态判断

已生成 `outputs/status_examples_10/`：来自 Nissan 和 Impala 两辆车的10种状态，
包含4个大偏移、4个小偏移、2个未扰动的正确框，不是10辆不同的车。
打开该目录的 `index.html` 可比较四视图，图片本身没有标题、图例或其他标注元素。
同车案例固定点云裁剪、相机裁剪及像素尺度，并检查红框没有被裁剪。

```bash
# 重新生成案例（目录存在时自动编号）。
python -m data_pipeline.create_status_examples

# 只做3选1状态判断，10例各调用一次API，不执行任何偏移。
python -m model_runtime.gemini_status_batch

# 可选：提供同车正确四图作为参考。
python -m model_runtime.gemini_status_batch --reference-mode same-target --output outputs/gemini_status_reference_10

# 可选：只导出请求，不联网。
python -m model_runtime.gemini_status_batch --dry-run --output outputs/status_preview
```

每例保存 `request.json`、`instructions.txt`、`response.json` 和 `decision.json`；
`results.json` 汇总预测、预期标签和置信度，`summary.json` 按偏移类别统计匹配数。
预期标签和偏移量仅供本地评估，不发给模型。同车参考模式有意发送正确框的图像。
预期标签以数据集原始标注为准：正确框预期接受，扰动框预期修正；
小偏移是否可以接受仍取决于容差。没有无效目标案例，因此这批数据不测试删除识别能力。

每轮现在还会保存 `before_state.json` 和 `comparison/` 下的四张前后框叠加图：
黄色虚线是本轮修改前的框，红色实线是修改后的框。每步对比的都是紧邻的前后状态。
接受、删除、停滞时几何不变，两框重合；删除状态仍在图中显示框供复核。
图片不含文字、坐标或图例。模型下一轮继续读取只有当前红框的 `views/`。
打开运行目录的 `index.html` 可按轮查看全部对比图；pipeline入口对应 `run/index.html`。

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
