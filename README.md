# TransferEEG

This is the code for the model "Multi-Source Structural Deep Clustering"

## Introduction

Unlike traditional domain adaption, the features in our model are not directly aligned from the source domain and the target domain due to significant discrepancy across subjects and sessions of EEG data, an optimal intermediate auxiliary distribution is developed by utilizing the predicted distribution of the target domain to be served as a destination that the target domain should move to. The features of the target domain are first aligned with the corresponding auxiliary domain. Then, during the joint training, the auxiliary distribution was replaced by the actual label distribution of the source domain to regularize the clustering. By employing this approach, the model in this paper are able to uncover the intrinsic structural distribution of the target domain itself without sacrificing the capacity to distinguish between different classes. The approach was also used to improve the distinguishability in the intermediate output space of the model. The predicted distribution is substituted in the initial method outlined above with the distribution of the intermediate features to the clustering centroid associated with each label.

## Architecture

![Figure 2](image/model.png)

Throughout the training phase, a common encoder is employed to extract fundamental features from both the source domains and the target domain, thereby establishing a shared foundation across all domains. Next, the target domain features are duplicated into multiple copies, with each copy corresponding to a specific source domain feature. Subsequently, an individual encoder and classifier are utilized for each source domain to extract features that are specific to that domain and perform the classification. In order to regularize the deep clustering in the intermediate feature space, ISCL is employed for the output from the private encoder. Additionally, CSCL is utilized to regularize the final deep clustering on the classification.

## Datasets

The datasets used are SEED dataset and SEED-IV dataset. The datasets can be downloaded from [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html).

The dataset format used for training is shown below.

```python
eeg_feature_smooth/
    1/
    2/
    3/
ExtractedFeatures/
    1/
    2/
    3/
```

## Usage

Relative configurations of the model can be adjusted in the `config.py`. To run the model, just run `python main.py`. The output of the model will be printed in the terminal

# Experiment result

| Dataset |  Model   | Cross Subject Accuracy (%) | Cross Subject F1-Score (%) | Cross-session Accuracy (%) | Cross-session F1-Score (%) |
| :-----: | :------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: |
|  SEED   |   DAN    |         68.71±7.12         |           68.05            |         79.86±8.72         |           79.21            |
|  SEED   |  MS-MDA  |         82.68±8.77         |           82.43            |         89.30±8.67         |           89.26            |
|  SEED   |  MS-ADA  |         86.16±7.87         |           86.13            |         91.10±7.08         |           91.05            |
|  SEED   |   UDDA   |         88.10±6.54         |           87.18            |        90.19±10.67         |           89.90            |
|  SEED   | **Ours** |       **90.69±9.28**       |         **90.65**          |       **95.05±6.08**       |         **95.02**          |
| SEED-IV |   DAN    |         41.71±8.25         |           42.68            |        54.64±11.63         |           55.61            |
| SEED-IV |  MS-MDA  |         65.66±9.18         |           65.61            |        67.70±13.71         |           67.64            |
| SEED-IV |  MS-ADA  |        66.56±10.24         |           66.53            |        66.68±11.86         |           67.43            |
| SEED-IV |   UDDA   |         73.14±9.43         |           73.01            |        75.96±11.55         |           75.87            |
| SEED-IV | **Ours** |      **74.35±12.06**       |         **74.31**          |      **78.56±12.81**       |         **78.53**          |