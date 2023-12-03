<!-- <br />

<div align="center">
  <h1 align="center">ICV-Net: Identity Cost Volume Network for Multi-View Stereo Depth Inference</h1>
  <p align="center">
    Pengpeng He<sup>1</sup></a>,
    Yueju Wang<sup>2</sup></a>,
    Yong Hu<sup>1</sup></a>,
    Wei, He<sup>1</sup></a>
    <br />
    1 Wuhan University of Technology; 2 Wuhan University
    <br />
    <br />
    <br />
  </p>

</div> -->


# ICV-Net: Identity Cost Volume Network for Multi-View Stereo Depth Inference

## Introduction

This is the official pytorch implementation of our paper: ICV-Net: Identity Cost Volume Network for Multi-View Stereo Depth Inference. In this work, we first propose novel identity cost volumes with the identical cost volume size at each stage, which dramatically decreases memory footprint while clearly improving depth prediction accuracy and inference speed. The depth inference is then formulated as a dense-to-sparse search problem that is solved by performing a classification to locate predicted depth values. Combining identity cost volumes with the dense-to-sparse search strategy, we propose an identity cost volume network for MVS, denoted as ICV-Net. The proposed ICV-Net is demonstrated on competitive benchmarks. Experiments show our method can reduce dramatically the memory consumption and extend the learned MVS to higher resolution scenes. 



## IMPORTANT NOTICE: 

<font face="微软雅黑" color=red size=5>The source code will be released once our paper is accepted. Stay tuned!</font>
