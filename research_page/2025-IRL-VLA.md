---
layout:  default
title:   "IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model"
permalink: /IRLVLA/
date: 2025-08-6 00:01:00 +0800
---

<div align="center">
<h3>IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model</h3>

Anqing Jiang<sup>1\*</sup>, Yu Gao<sup>1\*</sup>, Yiru Wang<sup>1\*</sup>, Zhigang Sun<sup>1</sup>, Shuo Wang<sup>1</sup>, Yuwen Heng<sup>1</sup>, <br>

Hao Jiang<sup>2</sup>，Jinhao Chai<sup>3</sup>, Zichong Gu<sup>3</sup>, Jijun Wang<sup>4</sup>, Li Sun<sup>5</sup>, Hao Sun†<sup>1</sup> <br>

<br>

<sup>1</sup>Bosch Corporate Research RIX  <br>
<sup>2</sup>Shanghai Jiaotong University  <br>
<sup>3</sup>Shanghai University <br>
<sup>4</sup>AIR, Tsinghua University <br>
<sup>5</sup>Robert Bosch GmbH <br>

<br>

(\*) Equal contribution. (†) Corresponding author.  <br>

</div>


## Abstract         
Vision-Language-Action (VLA) models have demonstrated potential in autonomous driving. However, two critical challenges hinder their development: (1) Existing VLA architectures are typically based on imitation learning in open-loop setup which tends to capture the recorded behaviors in the dataset, leading to suboptimal and constrained  performance, (2) Closed-loop training relies heavily on high-fidelity sensor simulation, where domain gaps and computational inefficiencies pose significant barriers. In this paper, we introduce IRL-VLA, a novel close-loop Reinforcement Learning via \textbf{I}nverse \textbf{R}einforcement \textbf{L}earning reward world model with a self-built VLA approach. Our framework proceeds in three-stage paradigm: In the first stage, we propose a VLA architecture and pretrain the VLA policy via imitation learning. In the second stage, we construct a lightweight reward world model via inverse reinforcement learning to enable efficient closed-loop reward computation. To further enhance planning performance, finally, we design specialized reward world model guidence reinforcement learning via PPO(Proximal Policy Optimization) to effectively balance the safety incidents, comfortable driving, and traffic efficiency. Our approach achieves state-of-the-art performance in NAVSIM v2 end-to-end driving benchmark, 1st runner up in CVPR2025 Autonomous Grand Challenge. We hope that our framework will accelerate VLA research in closed-loop autonomous driving.

```bibtex
@article{jiang2025diffvla,
  title={Diffvla: Vision-language guided diffusion planning for autonomous driving},
  author={Jiang, Anqing and Gao, Yu and Sun, Zhigang and Wang, Yiru and Wang, Jijun and Chai, Jinghao and Cao, Qian and Heng, Yuweng and Jiang, Hao and Dong, Yunda and others},
  journal={arXiv preprint arXiv:2505.19381},
  year={2025}
}
```