# Rethinking Classifier Re-Training in Long-Tailed Recognition: Label Over-Smooth Can Balance
## Environment set up
Install required packages:
```shell
conda create -n LOS python=3.7
conda activate LOS
pip install torch torchvision torchaudio
pip install progress
pip install pandas
```

## Run Experiments
### Stage1: Backbone Feature Learning
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
### Stage2: Classifier Re-Training
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
