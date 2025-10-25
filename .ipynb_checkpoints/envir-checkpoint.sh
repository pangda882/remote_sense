#!/bin/bash
set -e  # 出错即退出

echo "🔧 [1/10] 进入 home 目录"
cd ~
rm -rf miniconda3
echo "🌐 [2/10] 下载 Miniconda 安装脚本"
wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

echo "🛠️ [3/10] 添加执行权限"
chmod +x Miniconda3-latest-Linux-x86_64.sh

echo "📦 [4/10] 静默安装 Miniconda 到 ~/miniconda3"
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

echo "🧠 [5/10] 初始化 conda"
~/miniconda3/bin/conda init bash
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
echo "export CDPATH='/gemini/data-1/'" >> ~/.bashrc
source ~/.bashrc

echo "⚙️ [6/10] 下载 CUDA 11.8 安装包"
cd /usr/root
wget -nc https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

echo "🧾 [7/10] 安装 CUDA Toolkit（静默）"
chmod +x cuda_11.8.0_520.61.05_linux.run
./cuda_11.8.0_520.61.05_linux.run --silent --toolkit
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "📁 [8/10] 创建 conda 环境"
cd /gemini/code/ChangeCLIP
conda env create -f environment.yml


echo "🧱 [9/10] 安装 PyTorch 和依赖库"
conda activate changeclip
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm

echo "🔗 [10/10] 安装 OpenAI CLIP + MMCV"
cd /gemini/code
cd CLIP
pip install .
bash /gemini/code/ChangeCLIP/tools/clip_infer_sysu.sh

pip uninstall -y mmcv mmcv-full mmcv-lite
pip install mmcv==2.0.0rc4

echo "✅ 安装完成！请重新打开终端或执行 source ~/.bashrc 激活环境变量"
