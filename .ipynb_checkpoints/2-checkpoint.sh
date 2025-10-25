echo "📁 [8/10] 创建 conda 环境"
cd /gemini/code/ChangeCLIP
conda env create -f environment.yml

echo "🧱 [9/10] 安装 PyTorch 和依赖库"
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm

echo "🔗 [10/10] 安装 OpenAI CLIP + MMCV"
cd /gemini/code
cd CLIP
pip install .

pip uninstall -y mmcv mmcv-full mmcv-lite
pip install mmcv==2.0.0rc4

echo "✅ 安装完成！请重新打开终端或执行 source ~/.bashrc 激活环境变量"