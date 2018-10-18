# SLN-Text-Detection

## Scale-residual Learning Network for Text Detection in the wild.


## Introduction

<br>
This is a source code of the paper "Scale-residual Learning Network for Text Detection in the wild".
<br>

## Installation
<br>
    tensorflow version > 1.0
<br>
    python 2.7 or 3.5 
<br>

## Download
    The COCO-Text was used to train the SLN model for 10 epochs, and then training datasets including COCO-Text, ICDAR 2015, and UCAS-STLData were adopted to fine-tune the model until convergence. The SLN model evaluated on the minetto’s dataset used the model fine-tuned on ICDAR 2015. BaiduYun.
    
## Train
    If you want to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image and run Train_SLN.py.
    
## Test
    Run 
    <br>
    python eval_SLN.py 
    <\br>
    a text file will be then written to the output path.

## Results:
    In icdar 2015, F-score is 0.85 and FPS is 11.2. 
    In coco-text, F-score is 0.4661 ans FPS is 16.9. 
    In minetto's dataset, F-score is 0.81 ans FPS is 27.6.
    In UCAS_STLData, F-score is 0.91 ans FPS is 54.3.
    
    


