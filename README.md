# 自动标注：多视图框修正

当前仅保留两组结果：[最新Gemini修正](outputs/bench2drive_sedan_restart_001/before_after.png)和[Assistant/Sol对照](outputs/codex_sedan_refinement/before_after.png)。
其余旧实验和样例输出已清理，所需输入、场景信息和正确参考已分别归入保留结果的 `input/`。
重新运行请使用 [outputs/README.md](outputs/README.md) 中的新路径；下文旧样例命令需要先重新生成相应数据。


本项目独立放在 `auto_labeling/bev_box_refinement/`，与外层其他项目分开。
先切换到本目录，以下命令均在这里执行。依赖安装：

```bash
python -m pip install -r requirements.txt
```

也可以使用 Conda（配置使用 Python 3.11），在本目录执行：

```bash
conda env create -f environment.yml
conda activate auto_labeling
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

激活后，下面命令中的 `.venv/bin/python` 可统一替换为 `python`。
Git仓库与远程推送步骤见 [Git配置说明](docs/git-setup.md)。数据、参考图片、运行结果及API密钥不会提交。
当前代码通过 REST API 调用 Gemini，依赖为 NumPy、Matplotlib、Pillow、
laspy 和 LAZ 解压后端 lazrs，无需安装 PyTorch、CUDA 或 Google SDK。

## 代码目录

```text
data_pipeline/                  数据读取、初始框及测试扰动生成
  visualize_box_crops.py         读取 Bench2Drive，生成带标注的初始四视图和几何文件
  random_bev_offset.py           生成随机米制 XY / yaw 扰动及对比图
  create_small_offset.py         生成小像素扰动，导出125个动作

model_tools/                    模型决策执行时使用的工具
  geometry.py                   坐标变换、相机投影、三维框平移旋转
  actions.py                    动作词表、参数校验、动作执行、保留/删除状态
  render_views.py               将当前框渲染成无标题、坐标轴、图例的四视图

model_runtime/                  大模型运行入口
  prompts.py                    四视图说明、BEV像素约定和决策提示词
  gemini_box_decision.py         Gemini请求、响应解析、决策循环与结果保存
  box_refinement_pipeline.py    单框输入、四视图生成和独立轴修正流水线

tests/                          单元测试
docs/                           详细使用说明
data/bench2drive/               数据集入口（软链接）
outputs/                        图片、动作、几何及模型运行记录
requirements.txt                Python依赖
```

`data_pipeline` 调用 `model_tools` 生成示例；`model_runtime` 调用
`model_tools` 执行已选动作并重画图像。工具模块不依赖示例数据生成脚本。
现有数据及 `outputs/` 内文件路径保持不变。之前的独立脚本入口改为模块运行方式。

## 运行 Gemini

使用现有四张小偏移图片，先判断接受/修正/删除；需要修正时再分别选择X、Y和yaw的七选一动作：

```bash
export GEMINI_API_KEY='你的 API key'
.venv/bin/python -m model_runtime.gemini_box_decision --max-rounds 5 --output outputs/gemini_run_new
```

只检查请求、不调用API：

```bash
.venv/bin/python -m model_runtime.gemini_box_decision --dry-run --output outputs/request_check_new
```

输出目录若已存在，会自动追加 `_001`、`_002` 等编号，保留已有记录。
X、Y动作的像素单位以BEV俯视图为准，X向右、Y向下；
三维几何执行及其他视图重投影由工具负责。

## 数据生成与工具入口

```bash
# 读取数据、生成初始框与四视图。
.venv/bin/python -m data_pipeline.visualize_box_crops

# 基于现有参考像素映射，生成一个新的小偏移及125个动作。
.venv/bin/python -m data_pipeline.create_small_offset

# 生成小偏移后的四张纯净图片，保持参考裁剪与像素比例。
.venv/bin/python -m model_tools.render_views --offset outputs/bev_small_offset/offset.json --output outputs/small_offset_clean_views --reference-mapping outputs/offset_clean_views/pixel_mapping.json

# 测试。
.venv/bin/python -m unittest discover -s tests -v
```

详细说明：[数据与可视化](docs/visualization.md)、
[Gemini运行及视图约定](docs/gemini-decision.md)、[工具接口](docs/model-tools.md)。

## 十例状态判断

已生成4个大偏移、4个小偏移、2个正确框（两辆车的10种状态）：
[查看四视图](outputs/status_examples_10/index.html)、[偏移与预期标签](outputs/status_examples_10/manifest.json)。

```bash
python -m model_runtime.gemini_status_batch
```

每例只调用一次接受/修正/删除判断，不执行修正动作。可加 `--reference-mode same-target`
提供同车正确框的四视图参考。详见 [运行说明](docs/gemini-decision.md#十例状态判断)。

## 单框修正流水线

```bash
python -m model_runtime.box_refinement_pipeline \
  --box outputs/status_examples_10/case_03/state.json \
  --reference-example outputs/status_examples_10/case_05/views \
  --reference-mode same-target \
  --max-rounds 10 --output outputs/single_box_pipeline
```

输入框可为8×3 LiDAR角点数组、含 `corners_lidar` 的JSON，或已有 `state.json`。
需要匹配场景的 `--source` 和数据集；默认 source 为已有Nissan样例 `outputs/box_crops`。
使用其他车辆时必须指定对应source。无正确参考时省略两个reference参数即可。

每轮先接受/修正/删除；需要修正时，同一个请求分别询问X、Y、yaw，各7选1。
默认X/Y为 `[-10,-5,-1,0,1,5,10]` 原始BEV像素，yaw为 `[-5,-2,-1,0,1,2,5]` 度。
可设 `--pixel-step 10 --yaw-step 5`。先绕框中心旋转，再平移，然后更新四视图重新判断。
三个分量都为0时记录 `stalled` 并停止，接受与删除由第一阶段决定。
`input/` 保存输入视图；`run/final_state.json` 保存最终框，`run/history.json` 保存过程。
详见 [流水线说明](docs/box-pipeline.md)。

每轮现在还会保存 `before_state.json` 和 `comparison/` 下的四张前后框叠加图：
黄色虚线是本轮修改前的框，红色实线是修改后的框。每步对比的都是紧邻的前后状态。
接受、删除、停滞时几何不变，两框重合；删除状态仍在图中显示框供复核。
图片不含文字、坐标或图例。模型下一轮继续读取只有当前红框的 `views/`。
打开运行目录的 `index.html` 可按轮查看全部对比图；pipeline入口对应 `run/index.html`。

设置 `--max-rounds 100 --visualize-every 5` 可最多运行100轮，每5轮保存前后框对比。
提前结束和最后一轮也会保存；每轮模型使用的纯净四视图继续更新。
浏览页仅收录实际保存的对比轮次，每张比较该轮修改前后的相邻状态。

## nuScenes近车四视图

已生成 [nuScenes样例](outputs/nuscenes_box_crops/index.html)，使用本地nuScenes mini的原始车辆标注。
相机图与LiDAR分别使用各自时间戳的自车位姿和标定；nuScenes相机是光学坐标，不能套CARLA的轴变换。
此导出保留nuScenes LiDAR坐标，因此BEV=XY、正视=XZ、侧视=YZ。

```bash
python -m data_pipeline.visualize_nuscenes_box
```

也可用 `--sample-token`、`--box-token` 指定目标。`--data-root` 和 `--version` 可切换数据版本。
默认自动选择白天、前相机完整可见且较近的车辆；加 `--allow-night` 可包括夜间。
导出脚本不依赖nuScenes SDK。现有修正pipeline已支持以该导出目录作为nuScenes的source和正确参考。

### 官方nuScenes devkit相机可视化

独立脚本直接调用 `NuScenes.get_sample_data` 获取相机坐标下的框，
并调用官方 `Box.render` 绘制同一帧、同一辆车，默认生成完整相机图和同裁剪局部图。

```bash
conda run -n drivetransformer python -m data_pipeline.visualize_nuscenes_devkit_camera

# 同一前相机中全部可见标注框：
conda run -n drivetransformer python -m data_pipeline.visualize_nuscenes_devkit_camera --all-boxes
```

已有drivetransformer环境包含devkit。在其他环境使用前可安装 `requirements-nuscenes.txt`。
默认输入 `outputs/nuscenes_box_crops/metadata.json`，输出 `outputs/nuscenes_devkit_camera/`，
目录存在时自动编号。官方Box.render还会画一条表示框前方的短线；图片没有标题、坐标或图例。

### nuScenes随机偏移与修复

已生成 `outputs/nuscenes_random_offset/`，含 `offset.json`、`actions.json`、
模型输入 `views/` 和正确/偏移框对比 `comparison/`。默认随机平移20或30原始BEV像素，
yaw为正负2或5度，随机种子保存到状态，可用 `--seed` 重现。

```bash
python -m data_pipeline.create_nuscenes_offset

python -m model_runtime.box_refinement_pipeline \
  --box outputs/nuscenes_random_offset/offset.json \
  --source outputs/nuscenes_box_crops \
  --reference-example outputs/nuscenes_box_crops \
  --reference-mode same-target \
  --max-rounds 100 --visualize-every 5 \
  --output outputs/nuscenes_repair
```

无需指定data-root：nuScenes会读取source元数据中记录的数据集路径，Bench2Drive仍默认data/bench2drive。
按数据集使用相应的英文视图描述及坐标约定，BEV像素动作的正负定义保持一致。

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
