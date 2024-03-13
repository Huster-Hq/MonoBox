# MonoBox: Tightness-free Box-supervised Polyp Segmentation using Monotonicity Constraint

## 1. Overview
We propose MonoBox, an innovative box-supervised segmentation method constrained by monotonicity to liberate its training from the user-unfriendly box-tightness assumption. In contrast to conventional box-supervised segmentation, where the box edges must precisely touch the target boundaries, MonoBox leverages imprecisely-annotated boxes to achieve robust pixel-wise segmentation. The 'linchpin' is that, within the noisy zones around box edges, MonoBox discards the traditional misguiding multiple-instance learning loss, and instead optimizes a carefully-designed objective, termed monotonicity constraint. Along directions transitioning from the foreground to background, this new constraint steers responses to adhere to a trend of monotonically decreasing values. Consequently, the originally unreliable learning within the noisy zones is transformed into a correct and effective monotonicity optimization. Moreover, an adaptive label correction is introduced, enabling MonoBox to enhance the tightness of box annotations using predicted masks from the previous epoch and dynamically shrink the noisy zones as training progresses. 

<p align="center">
<img src="https://github.com/Huster-Hq/MonoBox/blob/main/Figs/framework.jpg" alt="Image" width="700px">
<p>

## 2. Results
### 2.1 Quantitative results
<p align="center">
<img src="https://github.com/Huster-Hq/MonoBox/blob/main/Figs/results0.png" alt="Image" width="700px">
<p>

### 2.2 Qualitative results
<p align="center">
<img src="https://github.com/Huster-Hq/MonoBox/blob/main/Figs/results1.png" alt="Image" width="700px">
<p>

We provide prediction resuts of all methods. You could download from [Google Drive](https://drive.google.com/drive/folders/19Au4OvsuBYyH0htpE8Xj_7drDvlZ30lB?usp=drive_link), including our results and that of compared methods on `Public Synthetic Noise Dataset` (ClinicDB, Kvasir-SEG, ColonDB, Endoscene, and ETIS).

## 3. Getting Started
### 3.1 Recommended environment:
```
Python 3.7.11
Pytorch 1.7.0
torchvision 0.8.0
```

### 3.2 Data preparation
Downloading training and testing datasets and move them into `./data/`, which can be found in this [Google Drive]()
```
MonoBox
├── data
├── ├── TrainDataset
├── ├── TestDataset
```

### 3.3 Pretrained model:
You could download the pretrained model from [Google Drive](https://drive.google.com/file/d/1Kc4utIDjBqquUKk6EfzTsrf0eBRhV7nH/view?usp=drive_link),  and then put it in the `./pretrained_pth` folder for initialization.
```
MonoBox
├── pretrained_pth
```

### 3.4 Test:
```
python Test.py
```
You could download the `well-trained model` from [Google Drive](https://drive.google.com/file/d/1Qi7tvsnm4bTTKYPPuLCPE12OQjeZ0SC1/view?usp=drive_link), and predict the results by `Test.py`.


### 3.4 Evaluation:
```
python eval.py
```
You could compute the evaluation metrics (Dice, IoU, and  HD) of the predictions.
