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

<br>
    The COCO-Text was used to train the SLN model for 10 epochs, and then training datasets including COCO-Text, ICDAR 2015, and UCAS-STLData were adopted to fine-tune the model until convergence. The SLN model evaluated on the minetto’s dataset used the model fine-tuned on ICDAR 2015.
<br>

## Train
        python train_SLN.py


## Test
        python eval_SLN.py 

## Results:
<br>
    In icdar 2015, F-score is 0.85. 
<br>
    In coco-text, F-score is 0.47.
<br>
    In minetto's dataset, F-score is 0.81.
<br>
    In UCAS_STLData, F-score is 0.91.
<br>
    The icdar 2013 and msra-td500 are being trained, and the evaluation results will be open.
    
    


