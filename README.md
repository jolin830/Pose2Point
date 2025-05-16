<p align="center">
  <h1 align="center">Pose2Point: Bridging Human Poses and Point Clouds for Unified Action Recognition</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=kU4rtNQAAAAJ&hl=en">Jiaying Lin</a></sup>,
    <a href="https://scholar.google.com/citations?user=goMNmRIAAAAJ&hl=en">Jiajun Weng</a></sup>,
    <a href="https://scholar.google.com.hk/citations?user=woX_4AcAAAAJ&hl=zh-CN">Mengyuan Liu</a></sup>
    <br>
    </sup> Peking University, Tsinghua University
  </p>
</p>

<div align=center>
<img src="https://github.com/jolin830/Pose2Point/tree/main/figs/P2P.png"/>
</div>


# Prerequisites
You can install all dependencies by running ```pip install -r requirements.txt```  <br />
Then, you need to install torchlight by running ```pip install -e torchlight```  <br />

# Data Preparation
## Download four datasets:
1. **NTU RGB+D 120** Skeleton dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/) <br />
2. **NTU RGB+D 60** Skeleton dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/) <br />



# Training and Testing
You can change the configuration in the yaml file and in the main function. We also provide four default yaml configuration files. <br />

## NTU RGB+D 120 dataset:
Cross-subject: run ```python main.py --device 0 --config ./config/NTU120_CS_default.yaml``` <br />
Cross-set: run ```python main.py --device 0 --config ./config/NTU120_CSet_default.yaml``` <br />


## NTU RGB+D 60 dataset:
Cross-subject, run ```python main.py --device 0 --config ./config/NTU60_CS_default.yaml``` <br />
Cross-view, run ```python main.py --device 0 --config ./config/NTU60_CV_default.yaml``` <br />




# Ensemble
Run ```./ensemble.py```
```
1. Perform 4 inferences to obtain the result pkl
2. Integrate these results
```

