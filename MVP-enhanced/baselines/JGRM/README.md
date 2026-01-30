# JGRM

Code of WWW'24 paper *More Than Routing: Joint GPS and Route Modeling for Refine Trajectory Representation Learning*.

A trajectory representation learning method that jointly models GPS points and routes for refined trajectory representations.

## Original Repository

**GitHub**: https://github.com/mamazi0131/JGRM

## Implementation Notes

本仓库中作为基线的 JGRM 实现与原始仓库完全一致，包括模型架构、超参数设置及训练流程，未做任何修改。

> The JGRM implementation used as a baseline in this repository is fully consistent with the original repository, including model architecture, hyperparameter settings, and training procedures. No modifications have been made.

## Citation

```bibtex
@inproceedings{10.1145/3589334.3645644,
  author = {Ma, Zhipeng and Tu, Zheyan and Chen, Xinhai and Zhang, Yan and Xia, Deguo and Zhou, Guyue and Chen, Yilun and Zheng, Yu and Gong, Jiangtao},
  title = {More Than Routing: Joint GPS and Route Modeling for Refine Trajectory Representation Learning},
  year = {2024},
  isbn = {9798400701719},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3589334.3645644},
  doi = {10.1145/3589334.3645644},
  booktitle = {Proceedings of the ACM on Web Conference 2024},
  pages = {3064–3075},
  numpages = {12},
  location = {Singapore, Singapore},
  series = {WWW '24}
}
```

## Requirements

- Python 3.7.12
- torch==1.7.1
- torch-geometric==2.3.1
- scikit-learn==1.0.2
- pandas==1.3.5
- pickleshare==0.7.5
- shapely==2.0.1
- faiss-cpu==1.7.4

## Dataset

Dataset is available at the original repository. See the GitHub link above for details.
