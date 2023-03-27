# IMP: Iterative Matching and Pose Estimation with Adaptive Pooling

<p align="center">
  <img src="assets/overview.png" width="960">
</p>

In this work, we propose to leverage global instances, which are robust to illumination and season changes for both
coarse and fine localization. For coarse localization, instead of performing global reference search directly, we search
for reference images from recognized global instances progressively. The recognized instances are further utilized for
instance-wise feature detection and matching to enhance the localization accuracy.

* Full paper PDF: [IMP: Iterative Matching and Pose Estimation with Adaptive Pooling](https://arxiv.org/abs/1911.11763).

* Authors: *Fei Xue, Ignas Budvytis, Roberto Cipolla*

* Website: [imp-release](https://github.com/feixue94/imp-release) for videos, slides, recent updates, and datasets.

## Dependencies

* Python==3.9
* PyTorch == 1.12
* opencv-contrib-python == 4.5.5.64
* opencv-python == 4.5.5.64

## Data preparation

Please download the training dataset of Megadepth following the instructions given
by [d2net](https://github.com/mihaidusmanu/d2-net/tree/dev). D2Net provided the preprocessed sfm models and undistorted
images on goole drive, but it is not working now, so you have to preprocess the Megadepth dataset on your own.

The data structure of Megadepth should be like this:

```
- Megadepth
 - phoenix
 - scene_info
    - 0000.0.npz
    - ...
 - Undistorted_SfM
    - 0000
        - images
        - sparse 
        - stereo
```

Then Use the command to extract local features (spp/sift), build correspondences for training:

```
python3 -m dump.dump_megadepth --feature_type spp --base_path  path_of_megadepth  --save_path your_save_path
```

The data structure of generated samples for training should like this:

```
- your_save_path
    - keypoints
        - 0000
            - 3409963756_f34ab1229a_o.jpg_spp.npy
    - matches
        - 0000
            - 0.npy
```

Instead of generating training samples offline, you can also do it online and adopt augmentations (e.g. perspective
transformation, illumination changes) to further improve the ability of the model.

## Training

Please modify <em> save_path </em> and <em> base_path </em> in configs/config_train_megadepth.json. Then start the
training as:

```
python3 train.py --config configs/config_train_megadepth.json
```

It requires 4 2080ti/1080ti gpus or 2 3090 gpus for batch size of 16.

## Results

1. Download the pretrained weights
   from [here](https://drive.google.com/drive/folders/1pI8_jnVhVX7BWa7M6H1s3GQgxrTwMncy?usp=sharing) and put them in
   the <em> weights </em> directory.

2. Prepare the testing data from YFCC and Scannet datasets. Download YFCC dataset:

```
   bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8
   tar -xvf raw_data_yfcc.tar.gz
```

Download preprocess Scannet evaluation data
from [here](https://drive.google.com/file/d/14s-Ce8Vq7XedzKon8MZSB_Mz_iC6oFPy/view)

3. Run the following the command for evaluation:

```
python3 -m eval.eval_imp --matching_method IMP --dataset yfcc
```

You will get results like this:

## BibTeX Citation

If you use any ideas from the paper or code from this repo, please consider citing:

```
@inproceedings{xue2022imp,
  author    = {Fei Xue and Ignas Budvytis and Roberto Cipolla},
  title     = {IMP: Iterative Matching and Pose Estimation with Adaptive Pooling},
  booktitle = {CVPR},
  year      = {2023}
}
```

## Acknowledgements

Part of the code is from previous excellent works
including [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), [SuperGlue]() and [SGMNet](). You can
find more details from their released repositories if you are interested in their works. 