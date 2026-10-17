# 推理环境记录

## 安装二维推理环境

在交付包根目录建立独立环境。以下命令以 Linux/macOS 的 Python 3.12 为例；
Windows 请使用对应的虚拟环境激活命令。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先安装相互匹配的 `torch` 和 `torchvision`。CPU 或 CUDA 的安装源取决于组员机器，
按 PyTorch 官方安装选择器给出的命令执行；不要盲目套用本机 CUDA wheel。
确认两者能够导入后，再安装随包固定的其余依赖及本包入口：

```bash
python -m pip install -r requirements-inference.txt
python -m pip install --no-deps -e .
python -m pip check
python -m infradetect_prototype --help
python -m pytest -q -p no:cacheprovider tests
python -m infradetect_prototype smoke --output /path/to/new_smoke_directory
```

`--no-deps` 避免安装入口时重新选择 PyTorch。固定版本是本机运行快照，不保证每种
操作系统都能取得相同 wheel；若安装冲突，应记录实际版本并重新跑回归测试。
没有 GPU 时可用 CPU 推理，但速度会慢得多；`torch.cuda.is_available()` 可检查
CUDA 是否可用。这个包不含训练代码，也不需要训练数据集。`smoke` 的
合成输入只验证随包权重与二维流程能够启动，不能评估缺陷准确率。

2026-10-17 的交付检查将**仅这个文件夹**复制到独立临时目录，并从该目录之外
运行 `smoke`：五份权重均加载，二维九阶段结束，`status.json` 为
`{"stage": "complete", "complete": true}`；该副本的 25 项测试通过。
这证明当前环境下没有运行时依赖原项目源码或训练数据，**不证明**其他机器
自动具备 Python 依赖，也不涵盖真实三维重建与模型准确率。

## 已检查的版本

2026-10-15 在本机 `DeepLabs+/.venv` 完成了二维代码的导入检查。以下是当时观察到的
版本，**不是**跨平台兼容性保证：

- 运行环境：Python 3.12，PyTorch 2.13.0，torchvision 0.28.0。
- 图像与数组：NumPy 2.5.2，opencv-python-headless 5.0.0.93，
  Pillow 12.3.0，SciPy 1.18.1。
- 模型与配置：transformers 4.57.6，segmentation-models-pytorch 0.3.4，
  timm 0.9.7，safetensors 0.8.0，PyYAML 6.0.3。
- 运行与测试：tqdm 4.70.1；pytest 9.1.1 仅用于回归测试。

## 三维可选环境

该环境**没有安装 `pycolmap`**。三维命令已检查源码，但没有在这个环境中执行。
需要三维投影/多视角核对的组员，还须在兼容环境中安装 `pycolmap`，准备 COLMAP
重建及对应场景的稠密点云；安装版本应与实际 COLMAP 数据格式核对。
`pycolmap` 没有写入二维依赖配置，避免给只做二维推理的组员引入不必要的安装问题。
在共享环境安装依赖前，先确认现有软件包和 GPU/CUDA 的兼容性。

推理代码不会联网下载预训练权重。当前集成的 5 个推理 checkpoint 已放在 `models/`
并与随包配置对应；替换模型时应同时替换匹配的配置，并单独重新验证。
