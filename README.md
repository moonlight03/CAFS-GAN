# CAFS-GAN
## Overview
This project includes source codes and pre-trained model of CAFS-GAN.
## Environment
```
natsort==8.3.1
numpy==1.22.3
opencv-python==4.6.0.66
pytorch==1.13.0
scipy==1.9.3
tqdm==4.64.1
```
## Datasets
* [SSAF](https://moonlight03.github.io/DSE-Net/). A high quality multi-style artistic font dataset.
* [Fonts](http://ilab.usc.edu/datasets/fonts). A computer-generated multi-color font dataset.
### Organization of data
```
CAFS-GAN 
│
└───data_training
|    │
|    └───subfolder1
|    |      image1.png
|    |      image2.png
|    |      ...
|    └───subfolder2
|    |      image1.png
|    |      image2.png
|    |      ...
|    ...
|    └───subfolder8
|    |      image1.png
|    |      image2.png
|    |      ...
└───data_testing
|     │
|     └───subfolder1
|     |      image1.png
|     |      image2.png
|     |      ...
|     └───subfolder2
|     |      image1.png
|     |      image2.png
|     |      ...
|     ...
|     └───subfolder9
|     |      image1.png
|     |      image2.png
|     |      ...
└───datasets
└───models
...
```


## Training
## BibTeX
```
@inproceedings{xxx2023xxx,
  title     = {Compositional Zero-Shot Artistic Font Synthesis},
  author    = {Li, Xiang and Wu, Lei and Wang, Changshuo and Meng, Lei and Meng, Xiangxu},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  year      = {2023},
  note      = {Main Track},
}
```
