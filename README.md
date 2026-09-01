# CoLC (CVPR 2026)

CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion
([Paper](https://arxiv.org/abs/2603.00682))


## Installation

### 1. Basic Installation

This code is based on [CoAlign](https://github.com/yifanlu0227/CoAlign), so I recommend you visit [CoAlign Installation Guide](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie) for details!

Or you can refer to [OpenCOOD data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [OpenCOOD installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare
data and install CoAlign. The installation is totally the same as OpenCOOD.

```
conda create -n colc python=3.7 
conda activate colc

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install spconv-cu116

pip install -r requirements.txt

git clone https://github.com/CatOneTwo/CoLC.git
cd CoLC
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace 
```


### 2. Dependent packages required by CoLC.


(1) Install pointnet2-ops follows [Pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch), if failed see this [issue](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174)

(2) Cupoch library
```
pip install cupoch
```


## Data Preparation
mkdir a `dataset` folder under CoAlign. Put your [V2X-Sim](https://ai4ce.github.io/V2X-Sim/download.html) in this folder. 
```
CoLC/dataset

. 
├── V2X-Sim-2.0
│   ├── sweeps
│   └── v2.0-mini
├── v2xsim2_info
│   ├── v2xsim_infos_test.pkl
│   ├── v2xsim_infos_train.pkl
│   └── v2xsim_infos_val.pkl
```

Note  *.pkl file in `v2xsim2_info` can be found in [Google Drive](https://drive.google.com/drive/folders/16_KkyjV9gVFxvj2YDCzQm1s9bVTwI0Fw?usp=sharing)


## Training and Inference

Please follow the steps in `train_colc.bash` to train each module in order and
use `test_colc.bash` for inference. Replace the model-loading paths in the Bash
files with your own pretrained or trained model paths as needed.

## Citation
```
@InProceedings{Han_2026_CVPR,
    author    = {Yushan, Han and  Hui, Zhang and Qiming, Xia and Yi, Jin and Yidong,Li},
    title     = {CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {2983-2992}
}
```

## Acknowlege

This project is impossible without the code of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [CoAlign](https://github.com/yifanlu0227/CoAlign)!
