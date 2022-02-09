# SkylineCF
code for paper Skyline of Counterfactual Explanations (CIKM 2021)

# Requirements
```
pip install numpy, pandas, pytorch, sklearn
```

# Getting Start

- The cf/ folder contains the a series of CF methods, includign skylineCF in our paper.
- data/ folder contains the datasets used.
- model folder consists of the interface for loading the pretrained model.
- util folder includes the interface for loading data, skyline operator used in our paper.
- weights folder covers the weights of pretrained model on three datasets.
- test folder comprise the how to generate the counterfactual explaantions for our method and baselines.
- mlp_* file show how to train the models. 

# How to Cite
```
@inproceedings{wang2021skyline,
  title={The Skyline of Counterfactual Explanations for Machine Learning Decision Models},
  author={Wang, Yongjie and Ding, Qinxu and Wang, Ke and Liu, Yue and Wu, Xingyu and Wang, Jinglong and Liu, Yong and Miao, Chunyan},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={2030--2039},
  year={2021}
}
```
