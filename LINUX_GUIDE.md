# Linux 运行简明指南

这个项目可以在 Linux 上跑，建议直接用源码 + conda 环境。

## 1. 创建环境

```bash
conda create -n vsr python=3.12 -y
conda activate vsr
```

如果你的 `conda` 本身有问题，先把它修好再继续。

## 2. 安装 PyTorch

CUDA 11.8 示例：

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
# 或者
pip install torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

如果你只想跑 CPU 版，就换成 CPU 对应安装方式。

## 3. 安装项目依赖

先进入仓库里的 `resources` 目录：

```bash
cd resources
```

然后安装依赖：

```bash
pip install -r requirements.txt
```

注意：

- `pywin32`、`pyreadline3`、`onnxruntime-directml` 这类是 Windows 相关包，Linux 下通常不用装。
- 如果 `av` 卡在源码编译，先单独处理它，不要整份依赖反复重试。



CLI：

```bash
python backend/main.py -i input.mp4 -o output.mp4
```

## 5. 切换字幕去除模式

常用模式：

- `sttn-auto`
- `sttn-det`
- `lama`
- `propainter`
- `opencv`

命令行切换：

```bash
python backend/main.py -i input.mp4 -o output.mp4 --inpaint-mode sttn-det
```

也可以在 `resources/config/config.json` 里改 `InpaintMode`。

## 6. 小提示

- Linux 会自动使用 `resources/backend/ffmpeg/linux_x64/ffmpeg`
- `DirectML` 主要是 Windows 用的，Linux 不用管
- 如果你要的是稳定运行，先跑 `lama` 或 `sttn-det` 再调别的模式
