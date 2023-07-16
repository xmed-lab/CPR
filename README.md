# Context-Aware Pseudo-Label Refinement for Source-Free Domain Adaptive Fundus Image Segmentation
This repository contains Pytorch implementation of our source-free unsupervised domain adaptation (SF-UDA) method with context-aware pseudo-label refinement (CPR).

![method](./figures/method.png "")
## Introduction
Context-Aware Pseudo-Label Refinement for Source-Free Domain Adaptive Fundus Image Segmentation MICCAI 2023

In the domain adaptation problem, source data may be unavailable to the target client side due to privacy or intellectual property issues. Source-free unsupervised domain adaptation (SF-UDA) aims at adapting a model trained on the source side to align the target distribution with only the source model and unlabeled target data. The source model usually produces noisy and context-inconsistent pseudo-labels on the target domain, i.e., neighbouring regions that have a similar visual appearance are annotated with different pseudo-labels. 
This observation motivates us to refine pseudo-labels with context relations. Another observation is that features of the same class tend to form a cluster despite the domain gap, which implies context relations can be readily calculated from feature distances. To this end, we propose a context-aware pseudo-label refinement method for SF-UDA. Specifically, a context-similarity learning module is developed to learn context relations. Next, pseudo-label revision is designed utilizing the learned context relations. Further, we propose calibrating the revised pseudo-labels to compensate for wrong revision caused by inaccurate context relations. Additionally, we adopt a pixel-level and class-level denoising scheme to select reliable pseudo-labels for domain adaptation. Experiments on cross-domain fundus images indicate that our approach yields the state-of-the-art results.

## Installation
Create the environment from the `environment.yml` file:
```
conda env create -f environment.yml
conda activate cpr
```
## Data preparation
* Download datasets from [here](https://drive.google.com/file/d/1B7ArHRBjt2Dx29a3A6X_lGhD0vDVr3sy/view). The data was already preprocessed by [BEAL](https://github.com/emma-sjwang/BEAL).

## Training
The following are the steps for the Drishti-GS (Domain1) to RIM-ONE-r3 (Domain2) adaptation.
* Download the source domain model from [here](https://drive.google.com/file/d/1eubjs4sw_EcIvsoJLEhIvZZSJq97gxQS/view?usp=sharing) or specify the data path in `./train_source.py` and then run `python train_source.py.`
* Save the source domain model into folder `./logs/source`.
* Download the initial pseudo label from [here](https://drive.google.com/file/d/1RLyWqRUT1_esUOnMk08dLAY2flfOaPnh/view?usp=sharing) or specify the model path and data path in `./generate_pseudo.py` and then run `python generate_pseudo.py`.
* Save the initial pseudo label into folder `./generate_pseudo`.
* Run `cd ./cpr` to go to the context-aware pseudo-label refinement folder. 
* Download the trained context-similarity model from [here](https://drive.google.com/file/d/1qOnRFM3gtdy5pgd6K5l7tsJ2VvJS7Orz/view?usp=sharing) or specify the model path, data path and pseudo label path in `./sim_learn.py` then run `python sim_learn.py`.
* Save the context-similarity model into folder `./log`.
* Download the refined psuedo label from [here](https://drive.google.com/file/d/1f2qPQdv-qAkitb81VO-Y184khYkAp01A/view?usp=sharing) or specify the context-similarity model path, data path and pseudo label path in `./pl_refine.py` then run `python pl_refine.py`.
* Save the refined pseudo label into folder `./log`
* Run `python train_target.py` to start the target domain training process.
## Result
cup: 0.7503 disc: 0.9503 avg: 0.8503 cup: 9.8381 disc: 4.3139 avg: 7.0760
## REFUGE to Drishti-GS adaptation
Follow the same pipeline as above, but run these commands to specify the new parameters:
```
python train_source.py --datasetS Domain4
python generate_pseudo.py --dataset Domain1 --model-file /path/to/source_model
python sim_learn.py --dataset Domain1 --model-file /path/to/source_model --pseudo /path/to/pseudo_label
python pl_refine.py --dataset Domain1 --weights /path/to/context_similarity_model --logt 5 --pseudo /path/to/pseudo_label
python train_target.py --dataset Domain1 --model-file /path/to/context_similarity_model --num_epochs 20
```

## Acknowledgement
We would like to thank the great work of the following open-source projects: [DPL](https://github.com/cchen-cc/SFDA-DPL), [AffinityNet](https://github.com/jiwoon-ahn/psa).