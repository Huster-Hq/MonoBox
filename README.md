# MonoBox: Tightness-free Box-supervised Polyp Segmentation using Monotonicity Constraint

## 1. Introduction

## 2. Framework Overview
![](https://github.com/Huster-Hq/MonoBox/blob/main/Figs/framework.jpg)

## 3. Results
### 3.1 Comparison with Anti-noise SOTAs
#### *Quantitative results
![](https://github.com/Huster-Hq/MonoBox/blob/main/Figs/results0.png)

#### *Qualitative results
![](https://github.com/Huster-Hq/MonoBox/blob/main/Figs/results1.png)
We provide prediction resuts of all methods. You could download from [Google Drive](https://drive.google.com/drive/folders/19Au4OvsuBYyH0htpE8Xj_7drDvlZ30lB?usp=drive_link)/[Baidu Drive](), including our results and that of compared methods on `Public Synthetic Noise Dataset`.



## 4. Getting Started
### 4.1 Recommended environment:
```
Python 3.7.11
Pytorch 1.7.0
torchvision 0.8.0
```

### 4.2 Recommended environment:
Downloading training and testing datasets and move them into ./data/, which can be found in this[Google Drive]()
```
MonoBox
 - data/
```

### 4.3 Pretrained model:
You could download the pretrained model from [Google Drive](https://drive.google.com/file/d/1Kc4utIDjBqquUKk6EfzTsrf0eBRhV7nH/view?usp=drive_link).

### 4.4 Testing:
```
cd MonoBox
python eval.py
```
