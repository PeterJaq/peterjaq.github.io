---
title:          "[Original Research Paper] SparseMeXt: Unlocking the Potential of Sparse Representations for HDMap Construction"
date:           2025-05-12 00:01:00 +0800
selected:       true
pub:            "IROS"
pub_date:       "2025"
category:       "hd-map construction"
abstract: >-
    Recent advancements in high-definition HD map construction have demonstrated the effectiveness of dense representations, which heavily rely on computationally intensive bird's-eye view BEV features. While sparse representations offer a more efficient alternative by avoiding dense BEV processing, existing methods often lag behind due to the lack of tailored designs. These limitations have hindered the competitiveness of sparse representations in online HD map construction. In this work, we systematically revisit and enhance sparse representation techniques, identifying key architectural and algorithmic improvements that bridge the gap with--and ultimately surpass--dense approaches. We introduce a dedicated network architecture optimized for sparse map feature extraction, a sparse-dense segmentation auxiliary task to better leverage geometric and semantic cues, and a denoising module guided by physical priors to refine predictions. Through these enhancements, our method achieves state-of-the-art performance on the nuScenes dataset, significantly advancing HD map construction and centerline detection. Specifically, SparseMeXt-Tiny reaches a mean average precision mAP of 55.5% at 32 frames per second fps, while SparseMeXt-Base attains 65.2% mAP. Scaling the backbone and decoder further, SparseMeXt-Large achieves an mAP of 68.9% at over 20 fps, establishing a new benchmark for sparse representations in HD map construction. These results underscore the untapped potential of sparse methods, challenging the conventional reliance on dense representations and redefining efficiency-performance trade-offs in the field.

cover: /assets/images/research/2025-sparsemext/overall_framework.png
authors:
- Anqing Jiang*
- Jinhao Chai
- Yu Gao
- Yiru Wang
- Zhigang Sun
- Hao Sun
- Lijuan Zhu
links:
  Project Page: SparseMeXT.github.io
  Paper: https://arxiv.org/abs/2505.08808
  Code: https://github.com/peterjaq/sparsemext
  
#Unsplash: https://unsplash.com/photos/sliced-in-half-pineapple--_PLJZmHZzk

---