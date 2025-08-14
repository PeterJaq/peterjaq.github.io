---
title:          "[Original Research Paper] DiffSemanticFusion: Semantic Raster BEV Fusion for Autonomous Driving via Online HD Map Diffusion"
date:           2025-08-03 00:01:00 +0800
selected:       true
pub:            "Arxiv"
pub_date:       "2025"
category:       "Diffusion"
abstract: >-
    Autonomous driving requires accurate scene understanding, including road geometry, traffic agents, and their semantic relationships. In online HD map generation scenarios, raster-based representations are well-suited to vision models but lack geometric precision, while graph-based representations retain structural detail but become unstable without precise maps. To harness the complementary strengths of both, we propose DiffSemanticFusion -- a fusion framework for multimodal trajectory prediction and planning. Our approach reasons over a semantic raster-fused BEV space, enhanced by a map diffusion module that improves both the stability and expressiveness of online HD map representations. We validate our framework on two downstream tasks: trajectory prediction and planning-oriented end-to-end autonomous driving. Experiments on real-world autonomous driving benchmarks, nuScenes and NAVSIM, demonstrate improved performance over several state-of-the-art methods. For the prediction task on nuScenes, we integrate DiffSemanticFusion with the online HD map informed QCNet, achieving a 5.1% performance improvement. For end-to-end autonomous driving in NAVSIM, DiffSemanticFusion achieves state-of-the-art results, with a 15% performance gain in NavHard scenarios. In addition, extensive ablation and sensitivity studies show that our map diffusion module can be seamlessly integrated into other vector-based approaches to enhance performance. 

cover: /assets/images/research/2025-diffsemfusion/2025_diffsemfusion.jpg
authors:
    - Zhigang Sun*
    - Yiru Wang$^{\dagger}$
    - Anqing Jiang*
    - Shuo Wang
    - Yu Gao
    - Yuwen Heng
    - Shouyi Zhang
    - An He
    - Hao Jiang
    - Jinhao Chai
    - Zichong Gu
    - Wang Jijun
    - Shichen Tang
    - Lavdim Halilaj
    - Juergen Luettin
    - Hao Sun
links:
  Project Page: https://github.com/SunZhigang7/DiffSemanticFusion
  Paper: https://arxiv.org/abs/2508.01778
  Code: https://github.com/SunZhigang7/DiffSemanticFusion
  
#Unsplash: https://unsplash.com/photos/sliced-in-half-pineapple--_PLJZmHZzk

---