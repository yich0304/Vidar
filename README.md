<p align="center">

  <h1 align="center">Scratch → FCOS3D → ViDAR: Pretraining Effects on BEVFormer</h1>
  <h3 align="center">
    <strong>Yicheng Zou</strong>
    ,
    <strong>Lucy Tan</strong>
    ,
    <strong>Zhifeng Zheng</strong>
  </h3>
  <h3 align="center"><a href="./assets/gsplatam-paper.pdf">Paper</a> | <a href="./assets/gsplatam-poster.pdf">Poster
  <div align="center"></div>
</p>
</h3>
    
## Environment Setup 
```bash
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c omgarcia gcc-6
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install --no-cache-dir ninja tensorboard==2.13.0 nuscenes-devkit==1.1.10 lyft-dataset-sdk==0.0.8 pandas==1.4.4 llvmlite==0.31.0 einops fvcore seaborn iopath==0.1.9 timm==0.6.13 typing-extensions==4.5.0 pylint ipython==8.12 numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0

python setup.py install

pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd bevformer
mkdir ckpts
cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```
## Download Data
Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download).
**Prepare nuScenes data**
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```
Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.
**Folder structure**
```bash
bevformer
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```
## Run and evaluate
### Train and test
Train BEVFormer
```
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 1
```
Eval BEVFormer
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./path/to/ckpts.pth 8
```
## Visualization 

see [visual.py](main/tools/analysis_tools/visual.py)
