# (MIBF-Net)Multisource joint representation learning fusion classification for remote sensing images

> Xueli Geng, Licheng Jiao, Lingling Li, Fang Liu, Xu Liu, Shuyuan Yang, Xiangrong Zhang
> *IEEE Transactions on Geoscience and Remote Sensing, 2023*

![fig24](img/framework.pdf)
## Abstract

Multisource remote sensing images provide complementary multidimensional information for reliable and accurate classification. However, gaps in imaging mechanisms result in heterogeneity between multiple source images. During fusion, this
heterogeneity causes the generated multisource representations to
be redundant and ignore discriminative unisource information,
which significantly hampers the fusion classification performance.
To address this challenge, we introduce a novel multisource joint
representation learning method for remote sensing image fusion
classification, termed multisource information bottleneck fusion
network (MIBF-Net). Based on the information bottleneck principle, MIBF-Net uses mutual information constraints to effectively
integrate multisource information, generating a comprehensive and nonredundant multisource representation. Specifically,
MIBF-Net first introduces an attribution-driven noise adaptation
layer to dynamically balance the speed of feature learning
across sources for extracting discriminative unisource intrinsic
information. Furthermore, a cross-source relationship encoding
(CRE) module is designed to fully explore cross-source complex
dependencies for enhancing the richness of fused representations. Finally, we design an information bottleneck fusion (IB-Fusion)module to fuse unisource semantic information and cross-source information while reducing redundancy. In particular, we use variational inference techniques to effectively address the mutual information optimization problem and provide theoretical derivations. Extensive experimental results on three heterogeneous multisource remote sensing data benchmarks show that the model significantly outperforms the state-of-the-art methods.
[[paper]](https://ieeexplore.ieee.org/abstract/document/10187157). 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Prerequisites

In order to run this implementation, you need to have the following software and libraries installed:

- Python 3.7
- PyTorch 1.3
- CUDA (if using GPU)
- NumPy
- Matplotlib
- OpenCV


### Training the Model

To train the model, you can run the following command:

```
python main.py
```

If you have any questions, please contact us (xlgeng@stu.xidian.edu.cn)



## Citation

Please cite our paper if you find this code useful for your research.

```
@ARTICLE{10187157,
  author={Geng, Xueli and Jiao, Licheng and Li, Lingling and Liu, Fang and Liu, Xu and Yang, Shuyuan and Zhang, Xiangrong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multisource Joint Representation Learning Fusion Classification for Remote Sensing Images}, 
  year={2023},
  volume={61},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2023.3296813}}
```
