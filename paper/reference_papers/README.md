# Reference papers: fusion and lightweight stereo matching

This folder contains local working copies of the papers selected for revising
the stereo-depth manuscript. They were copied from the user's curated research
library on 2026-09-27. These PDFs are research references, not project outputs;
do not add them to a public repository without confirming redistribution rights.

## Fusion / foundation-prior stereo

| PDF | Why it is included |
| --- | --- |
| `fusion/DEFOM-Stereo_Jiang_CVPR2025.pdf` | Combines a depth foundation model with recurrent stereo matching and explicit scale updates. |
| `fusion/D-FUSE_Yao_ICCV2025.pdf` | Aligns monocular priors with stereo using ordering representations and registered fusion. |
| `fusion/StereoAnywhere_Bartolomei_CVPR2025.pdf` | Uses complementary monocular and stereo cost volumes for difficult regions. |
| `fusion/MonSter_Cheng_CVPR2025.pdf` | Mutual refinement between monocular priors and multi-view/stereo geometry. |
| `fusion/AIO-Stereo_Zhou_AAAI2025.pdf` | Foundation-model knowledge distillation for deployment without a foundation model at inference. |
| `fusion/Fast-FoundationStereo_Wen_CVPR2026.pdf` | Compresses foundation-stereo knowledge toward real-time inference. |

## Lightweight / deployment-oriented stereo

| PDF | Why it is included |
| --- | --- |
| `lightweight/BGNet_Xu_CVPR2021.pdf` | Bilateral-grid cost-volume design. |
| `lightweight/CoEx_Bangunharcana_IROS2021.pdf` | Cost-volume excitation and aggregation for efficient stereo. |
| `lightweight/Distill-then-Prune_Pan_ICRA2024.pdf` | A direct reference for the teacher-to-student deployment strategy. |
| `lightweight/GGEV_Liu_AAAI2026.pdf` | Generalized geometry encoding volume, relevant to the StereoLite GEV branch. |
| `lightweight/HITNet_Tankovich_CVPR2021.pdf` | Tile/plane-based iterative stereo, relevant to the tile state and plane propagation. |
| `lightweight/LightStereo_Guo_ICRA2025.pdf` | Efficient stereo feature and aggregation design. |
| `lightweight/LiteAnyStereo_Jing_arXiv2025.pdf` | Lightweight zero-shot stereo direction. |
| `lightweight/MobileStereoNet_Shamsafar_WACV2022.pdf` | Mobile-oriented architectural baseline. |
| `lightweight/Pip-Stereo_Zheng_CVPR2026.pdf` | Iteration pruning for fast stereo refinement. |

Integrity check: run `sha256sum fusion/*.pdf lightweight/*.pdf` and compare
against the source library if needed.

## Joint semantic segmentation + stereo matching

| PDF | Why it is included |
| --- | --- |
| `semantic_stereo/S3M-Net_Wu_TIV2024.pdf` | Road-domain end-to-end semantic segmentation and iterative stereo matching with explicit feature fusion. |
| `semantic_stereo/RTS2Net_Dovesi_ICRA2020.pdf` | Compact real-time joint architecture with shared encoder, dual decoders, and semantic-guided disparity refinement. |
| `semantic_stereo/S3Net_Yang_IGARSS2024.pdf` | Single-branch joint semantic/disparity model with public US3D weights; satellite domain. |
| `semantic_stereo/SemStereo_Chen_AAAI2025.pdf` | Semantic-guided cascade, semantic-selective disparity refinement, and left-right semantic consistency; aerial domain. |
| `semantic_stereo/TiCoSS_Tang_TASE2025.pdf` | Road-domain tightly coupled joint model with official inference weights. |
| `semantic_stereo/DispSegNet_Zhang_RAL2019.pdf` | Early joint semantic/disparity residual-refinement architecture. |
| `semantic_stereo/SegStereo_Yang_ECCV2018.pdf` | Semantic feature embedding and semantic-consistency stereo regularisation. |
| `semantic_stereo/SSPCV-Net_Wu_ICCV2019.pdf` | Semantic/spatial pyramid cost volumes for stereo. |
| `semantic_stereo/SGNet_Chen_ACCV2020.pdf` | Semantic confidence and category-dependent stereo refinement. |
| `semantic_stereo/SDBF-Net_Rao_APSIPA2019.pdf` | Satellite semantic/disparity bidirectional fusion. |
| `semantic_stereo/USAM-Net_Sankaranarayanan_arXiv2025.pdf` | DrivingStereo semantic-attention depth estimation. |

Integrity check for this category: `sha256sum semantic_stereo/*.pdf`.
