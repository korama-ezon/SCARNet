# SCARNet: Spatial-Channel Attention and Multi-Scale Refinement Network for 3D Human Pose Estimation
This is the official implementation of the approach described in the paper:
> **SCARNet: Spatial-Channel Attention and Multi-Scale Refinement Network for 3D Human Pose Estimation**
> 
## Installation

SCARNet is tested on Ubuntu 24.10 with PyTorch 2.8.0 and Python 3.10. 

- Create a conda environment: 
  ```bash
  conda create -n scarnet python=3.10
  conda activate scarnet
- Install PyTorch 2.8.10 and Torchvision 0.23.0 following the official instructions
- Install dependencies:
  ```bash
  pip3 install -r requirements.txt
> 
## Dataset setup
Please refer to VideoPose3D to set up the Human3.6M dataset. Your directory structure should look like this:
```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_3d_3dhp.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
|   |-- data_2d_3dhp.npz
```
>
## Train the model
To train a 1-frame SCARNet model on Human3.6M from scratch:
```bash
# Train from scratch
python main.py --frames 1 --batch_size 256

# After training for 20 epochs, add refine module
python main.py --frames 1 --batch_size 256 --refine --lr 5e-4 --previous_dir [your best model saved path]
```
>
## Download pretrained model
The pretrained model can be found in <a href='https://drive.google.com/drive/folders/1n_UF1ZxxDCK9smKe6KwpCwVmMEoaDNi5?usp=drive_link'>here</a>, please download it and put it in the &#39;./checkpoint&#39; directory. 
>
## Test the model
To test a 1-frame SCARNet model on Human3.6M:
```bash
# Human3.6M
python main.py --test --previous_dir [your best model saved path] --frames 1

# MPI-INF-3DHP
python main.py --test --previous_dir [your best model saved path] --frames 1 --dataset '3dhp'
```

To test a 1-frame SCARNet model with refine module on Human3.6M:
```bash
python main.py --test --previous_dir [your best model saved path] --frames 1 --refine --refine_reload
```
## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
- [GraphMLP](https://github.com/Vegetebird/GraphMLP/tree/main)
## Licence

This project is licensed under the terms of the MIT license.
