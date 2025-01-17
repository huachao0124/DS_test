# bash Miniconda3-latest-Linux-x86_64.sh
# source /root/.bashrc
# conda install mkl
# pip install -U openmim
# mim install mmengine
# mim install "mmcv>=2.0.0"
mim install "mmcv>=2.0.0rc4,<2.2.0"
pip install -v -e .
pip install "mmsegmentation>=1.0.0"
pip install ftfy
pip install ood_metrics
pip install numpy==1.23.0
pip install nltk==3.8.1
python -m pip install ujson
echo 'export NLTK_DATA=./nltk_data' >> ~/.bashrc
source ~/.bashrc