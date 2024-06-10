# Revisiting Class-Incremental Object Detection: An Efficient Approach via Intrinsic Characteristics Alignment and Task Decoupling

## Introduction
>In real-world settings, object detectors frequently encounter continuously emerging object instances from new classes. Incremental Object Detection (IOD) addresses this challenge by incrementally training an object detector with instances from new classes while retaining knowledge acquired from previously learned classes. Despite recent advancements, existing studies reveal a critical gap: they diverge from the inherent characteristics of dense detectors, leaving considerable room for improving incremental learning efficiency. To address this challenge, we propose a novel and efficient IOD approach that aligns more closely with the intrinsic properties of dense detectors. Specifically, our approach introduces a learning-aligned mechanism, comprising tailored knowledge distillation and task alignment learning, to achieve more efficient incremental learning. Additionally, we propose expanding the classification network through task decoupling to alleviate performance limitations stemming from different optimization goals in the incremental learning process of the classification branch. Extensive experiments conducted on the MS COCO dataset demonstrate the effectiveness of our method, achieving state-of-the-art performance across various one-step and multi-step incremental scenarios. In multi-step incremental scenarios, our approach demonstrates a significant improvement of up to 12.9% in Average Precision (AP) compared to the previous method ERD.
![Alt text](doc/fig_pipeline.png "pipeline")
## Environments
- Python 3.8
- PyTorch 1.13.1
- CUDA 11.6
- mmdetection 3.0.0
- mmcv 2.0.0


**We are doing some preparation work and the code will be released soon.**
