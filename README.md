# IMP: Iterative Matching and Pose Estimation with Adaptive Pooling

<p align="center">
  <img src="assets/IMP.gif" width="960">
</p>

In this paper we propose an iterative matching and pose estimation framework (IMP) leveraging the geometric connections
between the two tasks: a few good matches are enough for a roughly accurate pose estimation; a roughly accurate pose can
be used to guide the matching by providing geometric constraints. To this end, we implement a geometry-aware recurrent
attention-based module which jointly outputs sparse matches and camera poses. Specifically, for each iteration, we first
implicitly embed geometric information into the module via a pose-consistency loss, allowing it to predict
geometry-aware matches progressively. Second, we introduce an efficient IMP, called EIMP, to dynamically discard
keypoints without potential matches, avoiding redundant updating and significantly reducing the quadratic time
complexity of attention computation in transformers.

With this code, you can train your own matcher from scratch with better performance than SuperGlue. As a trained model
supports different number of iterations (self/cross), you can choose a light version with fewer layers for easy tasks,
e.g., VO/SLAM and a heavy version with more layers for tough tasks such as long-term relocalization.

* Full paper PDF: [IMP: Iterative Matching and Pose Estimation with Adaptive Pooling](https://arxiv.org/abs/2304.14837).

* Authors: *Fei Xue, Ignas Budvytis, Roberto Cipolla*

* Website: [imp-release](https://feixue94.github.io/) for videos, slides, recent updates, and datasets.

## Dependencies

* Python==3.9
* PyTorch == 1.12
* opencv-contrib-python == 4.5.5.64
* opencv-python == 4.5.5.64

## Data preparation

Please download the preprocessed data of Megadepth (**scene_info** and **Undistorted_SfM**)
from [here](https://drive.google.com/drive/folders/1QJoXyrbYsk-ojrvGSgEpsGA0p5BVSGwZ?usp=sharing).

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
    - keypoints_spp
        - 0000
            - 3409963756_f34ab1229a_o.jpg_spp.npy
    - matches_spp # not used in the training process
        - 0000
            - 0.npy
    - matches_sep # this is used for loading data with multi-thread (tried h5py, but failed)
        - 0000
            - 0.npy
    - nmatches_spp # contains the number of valid matches (used for random sampling in the training process)
        - 0000_spp.npy 
    - mega_scene_nmatches_spp.npy # merged info of all scenes in nmatches_spp
```

Instead of generating training samples offline, you can also do it online and adopt augmentations (e.g. perspective
transformation, illumination changes) to further improve the ability of the model. Since this process is time-consuming
and there might be bugs in the code, it would be better to do a test of dumping and training on scenes in <em>
assets/megadepth_scenes_debug.txt </em>.

## Training

Please modify <em> save_path </em> and <em> base_path </em> in configs/config_train_megadepth.json. Then start the
training as:

```
python3 train.py --config configs/config_train_megadepth.json
```

The <em> base_path </em> in configs/config_train_megadepth.json should be the same as the <em> save_path </em> used
in <em> dump_megadepth </em>. It requires 4 2080ti/1080ti gpus or 2 3090 gpus for batch size of 16.

## Results

1. Download the pretrained weights
   from [here](https://drive.google.com/drive/folders/1QJoXyrbYsk-ojrvGSgEpsGA0p5BVSGwZ?usp=sharing) and put them in
   the <em> weights </em> directory.

2. Prepare the testing data from YFCC and Scannet datasets.

- Download YFCC dataset:

```
   bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8
   tar -xvf raw_data_yfcc.tar.gz
   
```

- Update the following entries in **dump/configs/yfcc_sp.yaml** and **dump/configs/yfcc_root.yaml**
    - [ ] **rawdata_dir**: path for yfcc rawdata
    - [ ] **feature_dump_dir**: dump path for extracted features
    - [ ] **dataset_dump_dir**: dump path for generated dataset
    - [ ] **extractor**: configuration for keypoint extractor

```
cd dump
python3 dump.py --config_path configs/yfcc_sp.yaml # copied from SGMNet
```

You will generate a hdf5 (**yfcc_sp_2000.hdf5**) file at **dataset_dump_dir**. Please also update the **rawdata_dir**
and **dataset_dir** in **configs/yfcc_eval_gm.yaml** and **configs/yfcc_eval_gm_sift.yaml** for evaluation.

- Download the preprocessed Scannet evaluation data
  from [here](https://drive.google.com/file/d/14s-Ce8Vq7XedzKon8MZSB_Mz_iC6oFPy/view)

- Update the following entries in **dump/configs/scannet_sp.yaml** and **dump/configs/scannet_root.yaml**
    - [ ] **rawdata_dir**: path for yfcc rawdata
    - [ ] **feature_dump_dir**: dump path for extracted features
    - [ ] **dataset_dump_dir**: dump path for generated dataset
    - [ ] **extractor**: configuration for keypoint extractor

```
cd dump
python3 dump.py --config_path configs/scannet_sp.yaml  # copied from SGMNet
```

You will generate a hdf5 (**scannet_sp_1000.hdf5**) file at **dataset_dump_dir**. Please also update the **rawdata_dir**
and **dataset_dir** in **configs/scannet_eval_gm.yaml** and **configs/scannet_eval_gm_sift.yaml** for evaluation.

3. Run the following the command for evaluation:

```
python3 -m eval.eval_imp --matching_method IMP --dataset yfcc
```

You will get results like this on YFCC dataset:

|       Model  | @5  | @10       | @20 |  
| -------- | -------- | ----- | ------- |  
| imp | 38.45 | 58.52 | 74.67  |  
| imp_iterative | 39.4 | 59.62    | 75.28  |   
|eimp | 36.96 | 56.76| 73.29 |  
|eimp_iterative | 38.98    | 58.95    | 74.81|  

## BibTeX Citation

If you use any ideas from the paper or code in this repo, please consider citing:

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
including [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
, [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
and [SGMNet](https://github.com/magicleap/SuperGluePretrainedNetwork). You can find more details from their released
repositories if you are interested in their works. 