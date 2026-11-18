# 创建独立Git仓库并推送

本项目的Git根目录是 `bev_box_refinement/`，不是外层 `auto_labeling/`。
版本管理只包含代码、测试、文档、依赖和空配置模板。
原始数据、`outputs/`、`examples/`、环境、缓存和API密钥均不提交。
忽略规则不会删除磁盘中的文件。

## 创建远程仓库

建议仓库名为 `bev-box-refinement`。在GitHub或你的Git托管平台创建空仓库，
可先选择Private；不要勾选自动生成README、gitignore或License，避免与本地首次提交产生分叉。
如团队已有GitLab，也可使用同样的推送步骤。

创建后复制仓库的SSH或HTTPS地址。在项目根目录执行以下步骤。
`YOUR_USERNAME` 和 `YOUR_REPOSITORY_URL` 是占位符，需要替换。

```bash
git remote add origin git@github.com:YOUR_USERNAME/bev-box-refinement.git
git push -u origin main
```

使用HTTPS也可以：

```bash
git remote add origin https://github.com/YOUR_USERNAME/bev-box-refinement.git
git push -u origin main
```

上面两种地址选一种，`remote add origin` 只执行一次。
如果已经配置过origin，先用 `git remote -v` 检查，需要修改时执行：

```bash
git remote set-url origin YOUR_REPOSITORY_URL
```

SSH方式需要账号已登记这台机器的SSH公钥；HTTPS方式使用托管平台支持的令牌或凭证管理器。
API key、Git访问令牌和私钥均不写入仓库文件或remote URL。

## 后续提交

```bash
git status
git diff
git add data_pipeline model_tools model_runtime tests docs README.md
git commit -m "Describe the change"
git push
```

首次配置作者身份时，使用你自己的真实姓名和邮箱，仅配置当前仓库即可：

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

克隆到其他机器后，重新创建环境、安装 `requirements.txt`、配置数据路径并生成输入样例。
`outputs/` 中的图片与实验结果不会随Git同步。
只有官方nuScenes devkit可视化脚本需要额外安装 `requirements-nuscenes.txt`。

## 本地API配置

```bash
cp .env.example .env
# 用编辑器填写.env中的GEMINI_API_KEY，再执行：
source .env
```

`.env.example` 是空模板，可以提交；`.env` 和 `.env.*` 的实际配置均被忽略。
