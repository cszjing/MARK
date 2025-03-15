# :zap: MARK
Modality Adaptation Representation Learning for Historical Document Image Retrieval


## 💡 Motivation

<p align="center">
    <img src="figs/fig1.jpg" alt="overview" width="800" />
</p>
Word spotting serves as a fundamental task in Historical Document Image Retrieval (HDIR) and can be categorized into query-by-example (QBE) and query-by-string (QBS) based on the query modality. QBS word spotting, as a cross-modal retrieval task, poses a significant challenge due to the substantial modality gap between image and text representations. Existing methods typically embed both image and text features into a common space and directly constrain their alignment, overlooking the substantial distribution gaps between modalities. Moreover, out-of-vocabulary (OOV) words present an additional challenge, requiring the model to possess strong generalization capability. To address these challenges, we propose a novel bi-directional modality adaptation representation learning framework, termed MARK. Specifically, the model adopts CLIP as the backbone to extract image and text features, providing a strong baseline for cross-modal representations. A bi-directional modality adaptation mechanism based on the Kolmogorov-Arnold network (KAN) is introduced to capture both shared and complementary information between image and text representations. Furthermore, Pyramid Histograms of Characters (PHOC) is introduced to enhance fine-grained text representation and overcome the limitations of CLIP's text encoder. Extensive experiments on benchmark datasets, including Kanjur and Geser, demonstrate that MARK establishes a new state-of-the-art in HDIR.

## 🤖 Architecture

<p align="center">
    <img src="figs/model.jpg" alt="overview" width="800" />
</p>

## 🔨 Installation
```bash
conda create -n diva python=3.9
conda activate mark
pip install -r requirements.txt
```
## 🌟 Main Results

## Comparative results of our MARK against existing methods on the Kanjur Dataset

| Method                   | Query | F1    | F2    | F3    | F4    | Avg   |
|--------------------------|-------|-------|-------|-------|-------|-------|
| BoVW [4]                 | **QBE** | 40.40 | 51.43 | 42.25 | 43.62 | 44.43 |
| AVWE [6]                 |       | 59.57 | 66.83 | 60.76 | 58.63 | 61.45 |
| RNN [6]                  |       | 54.54 | 62.05 | 53.30 | 60.82 | 57.68 |
| AVWE-SC [8]              |       | 63.25 | 65.06 | 63.99 | 64.71 | 64.25 |
| AVWE+RNN [6]             |       | 75.05 | 79.43 | 75.39 | 75.83 | 76.43 |
| Seq2seq (F2H1) [9]       |       | 80.16 | 85.08 | 79.08 | 84.59 | 82.23 |
| CNN (FC1) [10]           |       | 84.48 | 85.98 | 80.03 | 83.27 | 83.44 |
| Seq2seq+CNN [10]         |       | 88.02 | 90.35 | 84.62 | 87.21 | 87.55 |
| PUNet [14]               |       | 95.57 | 95.02 | 95.63 | 93.19 | 94.85 |
| EENet [HENet]            |       | 96.63 | 96.53 | 96.89 | 95.58 | 96.41 |
| HENet [HENet]            |       | 96.70 | 96.63 | 96.92 | 95.62 | 96.47 |
| CLIP* [clip]             |       | 97.99 | 98.52 | 98.36 | 97.99 | 98.21 |
| MetaCLIP* [metaclip]     |       | 98.90 | 99.17 | 99.20 | 98.76 | 99.01 |
| **MARK (Ours)**          |       | **99.10** | **99.40** | **99.27** | **98.89** | **99.16** |
|                          |       |        |        |        |        |        |
| PUNet [14]               | **QBS** | 88.46 | 85.67 | 91.77 | 90.11 | 89.00 |
| EENet [HENet]            |       | 88.90 | 94.93 | 92.61 | 92.39 | 92.21 |
| HENet [HENet]            |       | 89.19 | 95.13 | 93.21 | 92.51 | 92.51 |
| CLIP* [clip]             |       | 70.62 | 71.17 | 79.32 | 84.82 | 76.48 |
| MetaCLIP* [metaclip]     |       | 77.73 | 75.50 | 86.06 | 77.89 | 79.30 |
| **MARK (Ours)**          |       | **98.58** | **98.01** | **98.71** | **94.27** | **97.39** |

> *\* Indicates our reproduction, training on the Kanjur dataset using the loss function mentioned in Section 3.3.*




