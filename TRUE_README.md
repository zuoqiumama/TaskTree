# DISCO 复现与避坑指南

本文档基于实际 Debug 经验整理，旨在解决 AI2-THOR v2.1.0 在现代 Linux 服务器环境下的兼容性问题。

## 1. 环境安装 (关键)

由于 AI2-THOR 2.1.0 发布于 2019 年，它依赖非常旧的 Python Web 栈。**请务必严格按照以下版本安装**，否则会出现 `SocketException` 或 `ImportError`。

```bash
# 1. 创建环境 (Python 3.8 是必须的)
conda create -n disco python=3.8
conda activate disco

# 2. 安装依赖 (使用更新后的 requirements.txt)
pip install -r requirements.txt
```

验证安装：
运行 pip list | grep -E "ai2thor|Werkzeug|Flask|Jinja2"，确保版本如下：

ai2thor -> 2.1.0
Werkzeug -> 0.16.1
Flask -> 1.1.2
Jinja2 -> 2.11.3

## 2. 数据准备
下载 ALFRED 数据集（如果尚未下载）：
```bash
cd data
sh download_data.sh json
```

## 3.启动X server
打开一个新的终端窗口（Terminal 1）。
清理残留进程（防止冲突）：
```bash
sudo pkill Xorg
sudo rm -f /tmp/.X0-lock /tmp/.X11-unix/X0
```
再启动X server
```bash
python startx.py --gpu 0 --display 0
```
验证一下 X server是否存活：
```bash
export DISPLAY=:0
xset q
```
如果输出显示器配置信息（Monitor is On），说明 X Server 正常。如果报错 unable to open display，请检查 startx.py 是否报错。

如果出现乱七八糟的gdb的错误，把显卡换成别的

## 4.运行run.py!
打开另一个终端窗口，并清除代理
```bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```
然后运行run.py
```bash
python run.py --n_proc 1 --split valid_seen --x_display 0 --name debug_test --gpu []
```


## 5. 常见报错与修复

*Q1: SocketException: The socket has been shut down (Unity Log) / Timeout (Python)*
原因：Python 和 Unity 之间的 HTTP 通信被代理拦截，或者 Werkzeug 版本过高。
解决：
确保 pip install werkzeug==0.16.1。
确保运行前执行了 unset http_proxy ...。
检查 hosts 中是否有 127.0.0.1 localhost。

*Q2: ImportError: cannot import name 'escape' from 'jinja2'*
原因：Flask 1.1.2 依赖旧版 Jinja2，但环境中安装了新版（通常由 Jupyter 引入）。
解决：强制降级 pip install Jinja2==2.11.3 MarkupSafe==1.1.1。

*Q3: Exception: command ... exited with -9*
原因：X Server 崩溃或显存不足。
解决：
运行 nvidia-smi 检查显存。
执行“重启三部曲”：pkill Xorg -> startx -> run.py。