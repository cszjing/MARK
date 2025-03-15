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
conda create -n diva python=3.9

conda activate mark

pip install -r requirements.txt

