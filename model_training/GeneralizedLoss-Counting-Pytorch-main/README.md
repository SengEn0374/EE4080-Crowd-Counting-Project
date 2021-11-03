# A Generalized Loss Function for Crowd Counting and Localization

## Acknowledgement
The main codes are kindly made available by the authors at [Generalized_Loss](https://github.com/jia-wan/GeneralizedLoss-Counting-Pytorch)

## Data preparation
The UCF-QRNF dataset can be constructed followed by [Bayesian Loss](https://github.com/ZhihengCV/Bayesian-Crowd-Counting).
NWPU dataset can be constructed using the process_nwpu.py file.

## Pretrained model
The trained models can be downloaded with permission from [GoogleDrive](https://drive.google.com/drive/folders/1drinTf0G6LGF8Low9Yx0f2xX6rAbkkYB?usp=sharing).
The original pretrained model on UCF-QRNF can be accessed through the original author's repository above.

## Test

```
For inference on you own images, you will need to place your image(s) into inference/images folder, then run
python infer.py
Which will output an overlayed density map on your original image in inference/output

For testing on datasets, you will need to preprocess your own dataset with the relevant preprocess files made available here. You may modify the codes in cross-test.py to test on your desired dataset. 
python cross-test.py
```

## Train

```
python train.py --data-dir PATH_TO_DATASET --save-dir PATH_TO_CHECKPOINT
```

### Citation
If you use these code or models in your research, please cite the relevant authors with:

```
@InProceedings{Wan_2021_CVPR,
    author    = {Wan, Jia and Liu, Ziquan and Chan, Antoni B.},
    title     = {A Generalized Loss Function for Crowd Counting and Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    pages     = {1974-1983}
}
```

### Acknowledgement
We use [GeomLoss](https://www.kernel-operations.io/geomloss/) package to compute transport matrix. Thanks for the authors for providing this fantastic tool. The code is slightly modified to adapt to our framework.
