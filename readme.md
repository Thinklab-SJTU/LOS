# Rethinking Classifier Re-Training in Long-Tailed Recognition: Label Over-Smooth Can Balance (ICLR 2025)


## Authors: Siyu Sun*, Han Lu*, Jiangtong Li, Yichen Xie, Tianjiao Li, Xiaokang Yang, Liqing Zhang, Junchi Yan

## Abstract

In the field of long-tailed recognition, the Decoupled Training paradigm has shown exceptional promise by dividing training into two stages: representation learning and classifier re-training. While previous work has tried to improve both stages simultaneously, this complicates isolating the effect of classifier re-training. Recent studies reveal that simple regularization can produce strong feature representations, highlighting the need to reassess classifier re-training methods.  In this study, we revisit classifier re-training methods based on a unified feature representation and re-evaluate their performances. We propose two new metrics, Logits Magnitude and Regularized Standard Deviation, to compare the differences and similarities between various methods. Using these two newly proposed metrics, we demonstrate that when the Logits Magnitude across classes is nearly balanced, further reducing its overall value can effectively decrease errors and disturbances during training, leading to better model performance. Based on our analysis using these metrics, we observe that adjusting the logits could improve model performance, leading us to develop a simple label over-smoothing approach to adjust the logits without requiring prior knowledge of class distribution. This method softens the original one-hot labels by assigning a probability slightly higher than $\frac{1}{K}$ to the true class and slightly lower than $\frac{1}{K}$ to the other classes, where $K$ is the number of classes. Our method achieves state-of-the-art performance on various imbalanced datasets, including CIFAR100-LT, ImageNet-LT, and iNaturalist2018.

[[Paper Link]](https://openreview.net/forum?id=OeKp3AdiVO)


## Installation

### Environment set up
Install required packages:
```shell
conda create -n LOS python=3.7
conda activate LOS
pip install torch torchvision torchaudio
pip install progress
pip install pandas
```

### Run Experiments
#### Stage1: Backbone Feature Learning
**Training:**

```shell
python main_stage1.py --imb_ratio 100 --cur_stage stage1
```
- The parameter `--imb_ratio` can take on `10, 50, 100` to represent three different imbalance ratios.
- Other main parameters such as `--lr`, `--wd` can be tuned. 

**Testing:**
```shell
python evaluate.py --imb_ratio 100 --cur_stage stage1
```
- You can use `--pretrained_pth` to define the path of the pretrained model of stage1. Otherwise, we will use the pretrained 
optimal model with corresponding`imb_ratio` and `cur_stage` for default.

#### Stage2: Classifier Re-Training
**Training**:
```shell
python main_stage2.py --imb_ratio 100 --cur_stage stage2 --label_smooth 0.98
```
- The parameter `--imb_ratio` can be `10, 50, 100` to represent three different imbalance ratios.
- The parameter label smooth value `--label_smooth` can be modified.
- Other main parameters such as `--finetune_lr`, `--finetune_wd` can be tuned. 

**Testing**:
```shell
python evaluate.py --imb_ratio 100 --cur_stage stage2
```

## Reference

If you find our work useful, please consider citing the this paper: coming soon.
