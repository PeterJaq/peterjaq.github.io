---
title:          "[Original Research Paper] IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model"
date:           2025-08-12 00:01:00 +0800
selected:       true
pub:            "Arxiv"
pub_date:       "2025"
category:       "VLA"
abstract: >-
    Vision-Language-Action (VLA) models have demonstrated potential in autonomous driving. However, two critical challenges hinder their development: (1) Existing VLA architectures are typically based on imitation learning in open-loop setup which tends to capture the recorded behaviors in the dataset, leading to suboptimal and constrained performance, (2) Close-loop training relies heavily on high-fidelity sensor simulation, where domain gaps and computational inefficiencies pose significant barriers. In this paper, we introduce IRL-VLA, a novel close-loop Reinforcement Learning via Inverse Reinforcement Learning reward world model with a self-built VLA approach. Our framework proceeds in a three-stage paradigm: In the first stage, we propose a VLA architecture and pretrain the VLA policy via imitation learning. In the second stage, we construct a lightweight reward world model via inverse reinforcement learning to enable efficient close-loop reward computation. To further enhance planning performance, finally, we design specialized reward world model guidence reinforcement learning via PPO(Proximal Policy Optimization) to effectively balance the safety incidents, comfortable driving, and traffic efficiency. Our approach achieves state-of-the-art performance in NAVSIM v2 end-to-end driving benchmark, 1st runner up in CVPR2025 Autonomous Grand Challenge. We hope that our framework will accelerate VLA research in close-loop autonomous driving.

cover: /assets/images/research/2025-irlvla/2025_irlvla.png
authors:
    - Anqing Jiang*
    - Yu Gao*
    - Yiru Wang
    - Zhigang Sun
    - Shuo Wang
    - Yuwen Heng
    - Hao Sun$^{\dagger}$
    - Shichen Tang
    - Lijuan Zhu
    - Jinhao Chai
    - Jijun Wang
    - Zichong Gu
    - Hao Jiang
    - Li Sun
links:
  Project Page: https://github.com/IRL-VLA/IRL-VLA
  Paper: https://arxiv.org/abs/2508.06571
  Code: https://github.com/IRL-VLA/IRL-VLA
  
#Unsplash: https://unsplash.com/photos/sliced-in-half-pineapple--_PLJZmHZzk

---