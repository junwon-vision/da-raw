# DA-RAW: Domain Adaptive Object Detection for Real-world Adverse Weather Conditions (ICRA 2024)

This repository contains the code and data for **DA-RAW: Domain Adaptive Object Detection for Real-world Adverse Weather Conditions**, presented at ICRA 2024.

<div align="center">
    <a href="https://arxiv.org/pdf/2309.08152">Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://minsikjeon.github.io/DA-RAW/">Homepage</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://www.youtube.com/watch?v=yUert9RGyMc">Video</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://www.youtube.com/watch?v=vsUSrFsbuu8">Result Video</a>
    <br />
</div>

---

<p align="center">
  <img width="600" src="figs/concept_fig_compressed_figma.png" alt="Concept Figure">
</p>

## Table of Contents
- [Network](#network)
- [Using DA-RAW](#using-da-raw)
  - [Dependencies](#dependencies)
  - [Datasets](#datasets)
  - [Training and Testing](#training-and-testing)
- [Results](#results)
  - [BDD 100K & Custom Dataset](#bdd-100k--custom-dataset)
  - [Visualization](#visualization)
- [Acknowledgements](#acknowledgements)
- [Reference](#reference)

## Network: **DA-RAW**

<p align="center">
  <img width="600" alt="Network Diagram" src="figs/model_fig_compressed_figma.png">
</p>

## Using DA-RAW

### Dependencies
DA-RAW is based on the [Detectron2](https://github.com/facebookresearch/detectron2) framework. Follow the [Detectron2 docker installation](https://github.com/facebookresearch/detectron2/blob/main/docker/README.md) instructions to set up the environment.

Additionally, install the required packages listed in `requirements.txt`.

### Datasets

#### Cityscapes
- Download the Cityscapes dataset [here](https://www.cityscapes-dataset.com/).
- Convert the dataset to Pascal VOC format using the script provided [here](https://github.com/yuhuayc/da-faster-rcnn/tree/master), specifically the `prepare_data/prepare_data.m` script.
- Use all images in the `train` subset as the source dataset.

#### BDD 100K
- Download the BDD 100K dataset [here](https://www.vis.xyz/bdd100k/).
- Convert the dataset to Pascal VOC format and place it in the `detectron2/datasets/VOC2007` folder:
  - **Images**: Place in `JPEGImages/`.
  - **Annotations**: Place in `Annotation/`.
  - **Data Lists**: Place the lists of train (target domain) and test data in `ImageSets/Main/train_t.txt` and `test.txt`.
- To convert BDD 100K to Pascal VOC format, specify the dataset folder and run:
  ```
  cd datasets/
  python bdd_to_pascalvoc.py
  ```
  * As a result, you will get train_t and test files of each weather condition and Annotation, JPEGImages folder.

### Training/Testing
* To start a training using bdd rain dataset,
  ```
  cd tools/
  ./train_net_da_img.py --resume --num-gpus 4 --config-file ../configs/PascalVOC-Detection/FPN_DARAW_bdd_rain.yaml
  ```
  * Create or modify yaml file in Config directory.

* To test a model, simply add *--eval-only*, and specify MODEL.WEIGHTS you want to evaluate.
  ```
  ./train_net_da_img.py --eval-only --resume --num-gpus 4 --config-file ../configs/PascalVOC-Detection/FPN_DARAW_bdd_rain.yaml MODEL.WEIGHTS ./final/model_0011499.pth
  ```

#### YAML Parameters
- **MODEL**
  - **WEIGHTS**: *Path to pretrained weights*

  - **DA** *(Adversarial Alignment Settings)*:
    - **ADAPT_LEVEL**:  
      *Level of Alignment. Options: `'image'`, `'inst'`, `'imginst'`, `'all'` (including consistency loss)*
    - **CBAM**:  
      *Enable or disable CBAM (Convolutional Block Attention Module)*
    - **ALPHA**:  
      *Weight for the gradient reversal layer*
    - **DA_APPLY_LAYER**:  
      *Layers to attach the image-level style alignment module. Options: `['p2', 'p3', 'p4', 'p5']`*

  - **CL** *(Contrastive Learning Settings)*:
    - **MODE**:  
      *Mode of Contrastive Learning. Options: `None`, `'contrast'`, `'prototype'`, `'per_class_prototype'`*
    - **NUM_PROTOTYPE**:  
      *Number of prototypes per each class (or for the entire dataset)*
    - **CONTRAST_LAYER**:  
      *Layer to add instance-level weather alignment module. Options: `[0, 1, 2, 3]` where 0 is p2 ~ 3 is p5* 

- **DATASETS**
  - **TRAIN**: *Path to the source training dataset*

  - **TRAIN_TARGET**: *Path to the target training dataset*

  - **TEST**: *Path to the testing dataset*


## Results

#### BDD 100K & Our Dataset
| Method            | Source Data | Synthetic Data | Target Data | RainCityscapes (Synthetic) | Rain Rendering (Synthetic) | BDD 100K  | Our Dataset | BDD 100K | Our Dataset |
|-------------------|-------------|----------------|-------------|---------------------|-------------------|---------------|-------------|---------------|-------------|
| Source Only       | ✔           | ✗              | ✗           | 35.0                | 31.4              | 31.6          | 49.4        | 27.9          | 57.8        |
| Physics-based | ✔           | ✔              | ✗           | 40.5                | 41.8              | 22.1          | 35.1        | 18.1          | 42.0        |
| MPRNet       | ✔           | ✔              | ✔           | 37.7                | 46.9              | 12.8          | 38.6        | 12.3          | 41.8        |
| SADA         | ✔           | ✔              | ✔           | 38.7                | 40.1              | 29.1          | 48.2        | 27.6          | 53.5        |
| SWDA         | ✔           | ✗              | ✔           | 37.7                | 36.7              | 31.1          | 49.3        | 28.4          | 58.4        |
| Ours              | ✔           | ✗              | ✔           | 37.7                | 35.6              | **34.5**      | **51.2**    | **30.3**      | **62.6**    |


#### Visualization 
* To visualize the bounding box of specific model, use visualize_data.py with specifying config file and MODEL.WEIGHT
```
visualize_data.py --source 'dataloader' --config-file ../configs/PascalVOC-Detection/FPN_DARAW_bdd_rain.yaml SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS ./final/model_0011499.pth
```



## Acknowledgement
We utilize data and code from: 
- [1] [Detectron2](https://github.com/facebookresearch/detectron2)
- [2] [BDD 100K](https://www.vis.xyz/bdd100k/)
- [3] [Rethinking Semantic Segentation: A Prototype View](https://github.com/tfzhou/ProtoSeg)
- [4] [DA-Faster R-CNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master)

## Reference
If you find our work useful in your research work, consider citing [our paper](https://arxiv.org/abs/2309.08152)
```
@inproceedings{jeon2024raw,
  title={Da-raw: Domain adaptive object detection for real-world adverse weather conditions},
  author={Jeon, Minsik and Seo, Junwon and Min, Jihong},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
'''
