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

Please download the training dataset of Megadepth following the instructions provided by [SuperGlue](). The data
structure of Megadepth should be like this:

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

## Pretrained weights

We provide pretrained weights for local feature detection and extraction, global instance recognition for Aachen_v1.1
and RobotCar-Seasons datasets, respectively, which can be downloaded
from [here](https://drive.google.com/file/d/1N4j7PkZoy2CkWhS7u6dFzMIoai3ShG9p/view?usp=sharing)

## Localization with global instances

Once you have the global instance masks of the query and database images and the 3D map of the scene, you can run the
following commands for localization.

* localization on Aachen_v1.1

```
./run_loc_aachn
```

you will get results like this:

|          | Day  | Night       | 
| -------- | ------- | -------- |
| cvpr | 89.1 / 96.1 / 99.3 | 77.0 / 90.1 / 99.5  |
| post-cvpr | 88.8 / 95.8 / 99.2 | 75.4 / 91.6 / 100 |

* localization on RobotCar-Seasons

```
./run_loc_robotcar
```

you will get results like this:

|        | Night  | Night-rain       | 
| -------- | ----- | ------- |
| cvpr | 24.9 / 62.3 / 86.1 | 47.5 / 73.4 / 90.0  |
| post-cvpr | 28.1 / 66.9 / 91.8 | 46.1 / 73.6 / 92.5 |

## Training

If you want to retrain the recognition network, you can run the following commands.

* training recognition on Aachen_v1.1

```
./train_aachen
```

* training recognition on RobotCar-Seasons

```
./train_robotcar
```

## BibTeX Citation

If you use any ideas from the paper or code from this repo, please consider citing:

```
@inproceedings{xue2022imp,
  author    = {Fei Xue and Ignas Budvytis and Roberto Cipolla},
  title     = {Efficient Large-scale Localization by Global Instance Recognition},
  booktitle = {CVPR},
  year      = {2023}
}
```

## Acknowledgements

Part of the code is from previous excellent works
including [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), [R2D2](https://github.com/naver/r2d2)
, [HLoc](https://github.com/cvg/Hierarchical-Localization). You can find more details from their released repositories
if you are interested in their works. 