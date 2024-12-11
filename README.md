<div align="center">
<h1>MonoBox</h1>
<h3>MonoBox: Tightness-free Box-supervised Polyp Segmentation using Monotonicity Constraint</h3>
<br>
<a href="https://scholar.google.com/citations?user=rU2JxLIAAAAJ&hl=en">Qiang Hu</a><sup><span>1</span></sup>, Zhenyu Yi</a><sup><span>2</span></sup>, Ying Zhou</a><sup><span>1</span></sup>, Fan Huang</a><sup><span>3</span></sup>, Mei Liu<sup><span>4</span></sup>, Qiang Li<sup><span>1,&#8224;</span></sup>, <a href="https://scholar.google.com/citations?user=LwQcmgYAAAAJ&hl=en">Zhiwei Wang</a><sup><span>1, &#8224;</span></sup>
</br>

<sup>1</sup>  WNLO, HUST, <sup>2</sup>  SES, HUST, <sup>3</sup>  UIH, <sup>4</sup> HUST Tongji Medical College
<br>
(<span>&#8224;</span>: corresponding author)
</div>

## 1. Overview
We propose MonoBox, an innovative box-supervised segmentation method constrained by monotonicity to liberate its training from the user-unfriendly box-tightness assumption. Note that MonoBox is plug-and-play and <u>can improve the tolerance of any MIL-based box-supervised segmentation methods (e.g., [BoxInst](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html), [BoxLevelSet](https://arxiv.org/abs/2207.09055), [IBoxCLA](https://arxiv.org/abs/2310.07248), etc.) to tightness-free box annotations</u> by simply replacing origional box-supervised loss function (e.g., [MIL loss](https://github.com/chengchunhsu/WSIS_BBTP), [projection loss](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BoxInst/README.md), and [improved box-dice loss](https://arxiv.org/pdf/2310.07248)) with our proposed monotonicity constraint (MC) loss.

<p align="center">
<img src="https://github.com/Huster-Hq/MonoBox/blob/main/Figs/fig2.png" alt="Image" width="800px">
<p>

## 2. Targeted problem: tightness-free box annotations
Due to the carelessness of annotators and the blurred edges of target objects, the tightness-free box annotations are produced. The figure below shows some examples of non-tight box annotations produced by endoscopists in the real annotation process (the red dashed lines indicate regions where the annotation is too wide, and the yellow dashed lines indicate regions where the annotation is too narrow).
<p align="center">
<img src="https://github.com/Huster-Hq/MonoBox/blob/main/Figs/fig0.png" alt="Image" width="800px">
<p>

## 3. Results
### 3.1 Quantitative results
<p align="center">
<img src="https://github.com/Huster-Hq/MonoBox/blob/main/Figs/results0.png" alt="Image" width="700px">
<p>

### 3.2 Qualitative results
<p align="center">
<img src="https://github.com/Huster-Hq/MonoBox/blob/main/Figs/results1.png" alt="Image" width="700px">
<p>

We provide prediction resuts of all methods. You could download from [Google Drive](https://drive.google.com/drive/folders/19Au4OvsuBYyH0htpE8Xj_7drDvlZ30lB?usp=drive_link), including our results and that of compared methods on `Public Synthetic Noise Dataset` (ClinicDB, Kvasir-SEG, ColonDB, Endoscene, and ETIS).

## 4. Getting Started
### 4.1 Recommended environment:
```
Python 3.7.11
Pytorch 1.7.0
torchvision 0.8.0
```

### 4.2 Data preparation
Downloading training (with ground truth masks) and testing datasets and move them into `./data/`, which can be found in this [Polyp-PVT Repositories](https://github.com/DengPingFan/Polyp-PVT?tab=readme-ov-file) (due to dataset permission issues, we recommend downloading the dataset at this link).
The synthetic box annotations (sigma=0.2) generated by us could found in [Google Drive](https://drive.google.com/drive/folders/1fl1HkqPz1pwxwcr5N43vINg0v1jJmSU_?usp=drive_link), where .txt files record the class (all zeros), center point abscissa, center point ordinate, width, and height of synthetic noise boxes. The file paths should be arranged as follows:
```
MonoBox
├── data
├── ├── TrainDataset
├── ├── ├── images
├── ├── ├── ├── ├── 1.png
├── ├── ├── ├── ├── 2.png
├── ├── ├── ├── ├── ......
├── ├── ├── noisy_labels
├── ├── ├── ├── sigma_0.2
├── ├── ├── ├── ├── 1.txt
├── ├── ├── ├── ├── 2.txt
├── ├── ├── ├── ├── ......
├── ├── TestDataset
```

### 4.3 Pretrained model:
You could download the pretrained model from [Google Drive](https://drive.google.com/file/d/1Kc4utIDjBqquUKk6EfzTsrf0eBRhV7nH/view?usp=drive_link),  and then put it in the `./pretrained_pth` folder for initialization.
```
MonoBox
├── pretrained_pth
```

### 4.4 Training:
```
cd MonoBox
python Train.py
```

### 4.5 Testing:
```
python Test.py
```
You could also download the `well-trained model` from [Google Drive](https://drive.google.com/file/d/1Qi7tvsnm4bTTKYPPuLCPE12OQjeZ0SC1/view?usp=drive_link), and predict the results by `Test.py`.


### 4.6 Evaluation:
```
python eval.py
```
You could compute the evaluation metrics (Dice, IoU, and  HD) of the predictions, you could download prediction resuts directly (see in Sec. 2.2) and verify the performance.

## 5. Limitations
In the below figure, we show some failure cases of our methodon on the COCO dataset, where we combined our method with [BoxInst](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BoxInst/README.md) (a classical box-supervised instance segmentation model proposed for general objects). For clearer visualization, we use dashed boxes to indicate failure regions. These cases can show the limitations of our proposed method (i.e., MonoBox), mainly for `small-size/thin` objects or `non-connected` objects (e.g. `occluded` objects).

Importantly!!! Please note that these limited conditions almost never appear in colorectal endoscope polyp images, so our method is still very effective for the task that is the main focus of our paper, i.e., polyps. Here, we present these failure cases in order to facilitate readers to better understand our method and use our method in suitable scenarios.

<p align="center">
<img src="https://github.com/Huster-Hq/MonoBox/blob/main/Figs/Failure_cases.png" alt="Image" width="1200px">
<p>

## Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :

```BibTeX
@article{hu2024monobox,
  title={MonoBox: Tightness-free Box-supervised Polyp Segmentation using Monotonicity Constraint},
  author={Hu, Qiang and Yi, Zhenyu and Zhou, Ying and Li, Ting and Huang, Fan and Liu, Mei and Li, Qiang and Wang, Zhiwei},
  journal={arXiv e-prints},
  pages={arXiv--2404},
  year={2024}
}
```
