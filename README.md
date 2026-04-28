# Generating Tabular Synthetic Data

#### **Table of Contents**

[TOC]

# Executive Summary

##### **Problem context**

Advanced analytics is transforming all industries and is inherently data hungry. In health care data privacy rules detract data sharing for collaboration. Synthetic data, that retains the original characteristics and model compatible, can make data sharing easy and enable analytics for health care data.

##### **Need for change**

Conventionally statistical methods have been used, but with limited success. Current deidentification techniques are not sufficient to mitigate re-identification risks. Emerging technologies in Deep Learning such as GAN are very promising to solve this problem.

##### **Key Question**

How can you certify that the generated data is as similar and as useful as original data for the intended uses?

##### **Proposed Solution**

The Proposed solution involves generating Synthetic Data using Generative Adversarial Networks or GANs and with the help of conventionally available sources such as TGAN and CTGAN. 

The project also includes modules which can test the generated synthetic data against the original datasets on following three areas:

- **Statistical Similarity:** Create standardized modules to check if the generated datasets are similar to the original dataset 
- **Privacy Risk:** Create standardized metrics to check if generated synthetic data protects privacy of all data points in the original dataset.
- **Model Compatibility**: Compare performance of Machine Learning techniques on original and synthetic datasets

##### **Results**:

- **Statistical Similarity:** The generated datasets were similar to each other when compared using PCA and Auto-Encoders.
- **Privacy module**: The generated Privacy at Risk (PaR) metric and modules helps identify columns which are at risk to expose the privacy of data points from original data. The generated datasets using TGAN and CTGAN had sufficiently high privacy score and protected privacy of original data points.
- **Model Compatibility**: The synthetic data has comparable model performance for both classification and regression problems. 

# Introduction

This project deals with sensitive healthcare data that has Personal identifiable Information (PII) of 100M+ people. The healthcare industry is particularly sensitive as Patient Identifiable Information data is strictly regulated by the Health Insurance Portability and Accountability Act (HIPPA) of 1996. Healthcare firms need to keep customer data secure while leveraging it to innovate research and drive growth in the firm. However, current data sharing practices (to ensure de-identification) have resulted in wait times for data access as long as 3 months. This has proved to be a hindrance to fast innovation. The need of the hour is to reduce the time for data access and enable innovation while protecting the information of patients. The key question to answer here is:

"How can we safely and efficiently share healthcare data that is useful?"

##### **Complication**

The key questions involve the inherent trade-off between safety and efficiency. With the inception of big data, efficiency in the data sharing process is of paramount importance. Availability and accessibility of data ensure rapid prototyping and lay down the path for quick innovation in the healthcare industry. Efficient data sharing also unlocks the full potential of analytics and data sciences through use cases like the diagnosis of cancer, predicting response for drug therapy, vaccine developments, drug discovery through bioinformatics. Apart from medical innovation, efficient data sharing helps to bridge the shortcomings in the healthcare system through salesforce effectiveness, managing supply chain and improve patient engagement. While efficient data sharing is crucial, the safety of patient's data can not be ignored. Existing regulations like HIPPA and recent privacy laws like the California Consumer Privacy Act are focused on maintaining the privacy of sensitive information. More advanced attacks are being organized by hackers and criminals aimed at accessing personal information. As per IBM's report on cost data breaches, the cost per record is ~$150. But the goodwill and trust lost by the companies, cannot be quantified So, the balance between data sharing and privacy is tricky.

# History of the Project

Existing de-identification techniques involve two main techniques 1) Anonymization Techniques 2) Differential Privacy. Almost every firm relies on these techniques to deal with sensitive information in PII data. These techniques have proven to be successful in the past and thus act as low hanging fruit for any organization.

1. **Anonymization techniques:** These techniques try to remove the columns which contain sensitive information. Methods include deleting columns, masking elements, quasi-identifiers, k-anonymity, l-diversity, and t-closeness.

2. **Differential privacy:** This is a perturbation technique which adds noise to columns which introduce randomness to data and thus maintain privacy. It is a mechanism to help to maximize the aggregate utility of databases ensuring high levels of privacy for the participants by striking a balance between utility and privacy.

However, these techniques are not cutting edge when it comes to maintaining privacy and data sharing. Rocher et al have proven that 99.98 percent of Americans (in a sample size of the population of Massachusetts) would be correctly re-identified in any dataset using as few as 15 demographic attributes. They conclude that "even heavily sampled anonymized datasets are unlikely to satisfy the modern standards for anonymization set forth by GDPR and seriously challenge the technical and legal adequacy of the de-identification release-and-forget model."

**Proposition**

Currently, the field of AI which is being given a lot of importance is Deep Learning. It addresses the critical aspect of data science in this age through universality theorem (identifying function form) and representation learning (correct features). Of late, generative modeling has seen a rise in popularity. In particular, a relatively recent model called Generative Adversarial Networks or GANs introduced by Ian Goodfellow et al. shows promise in producing realistic samples. While this is a state-of-the-art deep learning models to generate new synthetic data, there are few challenges which we need to overcome.

| Salient Features                                         | Challenges                                                   |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| Neural Network is cutting edge algorithm in industry     | Trained to solve one specific task, can it fit all use cases? |
| Generate image using CNN architecture                    | Can we generate table from relational databases?             |
| Generate fake images of human faces that looks realistic | Would it balance the trade-off between maintaining utility and privacy of data |
| Requires high computational infrastructure like GPUs     | How to implement GAN for big data?                           |

# Why GANs?

## Introduction

A generative adversarial network (GAN) is a class of machine learning systems invented by Ian Goodfellow in 2014. GAN uses algorithmic architectures that use two neural networks, pitting one against the other (thus the "adversarial") in order to generate new, synthetic instances of data that can pass for real data.

GANs consist of Two neural networks contest with each other in a game. Given a training set, this technique learns to generate new data with the same statistics as the training set. The two Neural Networks are named Generator and a Discriminator.

## GAN Working Overview

**Generator**
The generator is a neural network that models a transform function. It takes as input a simple random variable and must return, once trained, a random variable that follows the targeted distribution. The generator randomly feeds actual image and generated images to the Discriminator. The generator starts with Generating random noise and changes its outputs as per the Discriminator. If the Discriminator is successfully able to identify that generate input is fake, then then its weights are adjusted to reduce the error.

**Discriminator**
The Discriminators job is to determine if the data fed by the generator is real or fake. The discriminator is first trained on real data, so that it can identify it to acceptable accuracy. If the Discriminator is not trained properly, then it in turn will not be accurately able to identify fake images thus poorly training the Generator.

This is continued for multiple iterations till the discriminator can identify the real/fake images purely by chance only.

**Algorithm:**
Now lets see how GANs algorithm works internally.

- The generator randomly feeds real data mixed with generated fake data for the discriminator
- To begin, in first few iterations, the generator produces random noise which the discriminator is very good at detecting that the produced image is fake.
- Every iteration, the discriminator catches a generated image as fake, the generator readjusts its weights to improve itself. much like the Gradient Descent algorithm
- Over time, after multiple iterations, the generator becomes very good at producing images which can now fool the discriminator and pass as real ones.
- Now, its discriminators turn to improve its detection algorithm by adjusting its network weights.
- This game continues till a point where the discriminator is unable to distinguish a real image from fake and can only guess by chance.

## Existing Research in Synthetic Data Generation

### TGAN

This methodology has been created from the work provided in this paper:
[Synthesizing Tabular Data using Generative Adversarial Networks](https://arxiv.org/pdf/1811.11264.pdf)

Tabular GAN (TGAN) is a generative adversarial network which can generate tabular data by learning distribution of the existing training datasets and can generate samples which are using the power of deep neural networks.

TGAN focuses on generating tabular data with mixed variable types (multinomial/discrete and continuous). To achieve this, we use LSTM with attention in order to generate data column by column. To asses, we first statistically evaluate the synthetic data generated by TGAN.

#### Implementation

```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import tensorflow as tf
from tgan.model import TGANModel
from tgan.data import load_demo_data

def tgan_run(data, cont_columns):
    tgan = TGANModel(continuous_columns)
    return tgan.fit(data)

def tgan_samples(model, num_samples):
    return tgan.sample(100000)
```

### CTGAN

CTGAN is a GAN-based method to model tabular data distribution and sample rows from the distribution. CTGAN implements mode-specific normalization to overcome the non-Gaussian and multimodal distribution. We design a conditional generator and training-by-sampling to deal with the imbalanced discrete columns.

**Several unique properties of tabular data challenge the design of a GAN model.**

+ Mixed data types: Real-world tabular data consists of mixed types.
+ Non-Gaussian distributions: Continuous values in tabular data are usually non-Gaussian where min-max transformation will lead to vanishing gradient problem.
+ Multimodal distributions: Vanilla GAN couldn't model all modes on a simple 2D dataset.
+ Learning from sparse one-hot-encoded vectors.
+ Highly imbalanced categorical columns.

#### Implementation

```python
import pandas as pd
import tensorflow as tf

from ctgan import load_demo
from ctgan import CTGANSynthesizer

data = load_demo()
discrete_columns = ['workclass','education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country', 'income']
ctgan = CTGANSynthesizer()
ctgan.fit(data, discrete_columns)
```

### Differentially Private GAN

Source: [https://arxiv.org/pdf/1802.06739.pdf](https://arxiv.org/pdf/1802.06739.pdf)

One common issue in GANs is that they can easily remember training samples. Differentially Private GANs is achieved by adding carefully designed noise to gradients during the learning procedure.

### PATE-GAN

Source: [https://arxiv.org/pdf/1906.09338.pdf](https://arxiv.org/pdf/1906.09338.pdf)

PATE GAN consists of two generator blocks called student block and teacher block on top of the existing generator block. PATE GAN prevents reconstruction by breaking down the generator into three stages.

### G-PATE

Compared to PATE-GAN, G-PATE improves the use of privacy budget by applying it to the part of the model that actually needs to be released for data generation.

# Methodology

In order to validate the efficacy of GANs to serve our purpose, we propose a methodology for thorough evaluation of synthetic data generated by GANs.

## MIMIC - III Dataset

MIMIC-III stands for Medical Information Mart for Intensive Care Unit. It is an open source database and contains multiple tables which can be combined to prepare various use cases.

Overview of MIMIC-III data:
- 26 tables
- 45,000+ patients
- 65,000+ ICU admissions
- PII data

## Use Case 1: Length of Stay

#### Goal
Predicting the length of stay in ICU using 4 out of 26 tables.

#### Final Data
- Level of data: Unique admission for each patient.
- Target Variables: 'LOS' (length of stay).
- Predictor variables: 18 columns of different diagnosis category.
- Other descriptive variables: "ADMISSION_TYPE", "MARITAL_STATUS","INSURANCE", "ETHNICITY", "HOSPITAL_EXPIRE_FLAG", "GENDER" and "EXPIRE_FLAG".

## Use Case 2: Mortality Prediction

#### Goal
Predict mortality just by the number of interactions between patient and hospital as predictors through count of lab tests, prescriptions, and procedures.

# Synthetic Data Generation

A synthetic dataset is a repository of data that is generated programmatically.

- It can be numerical, binary, or categorical.
- The number of features and length of the dataset should be arbitrary.
- The underlying random process can be precisely controlled and tuned.

##### Configuring GPU for Tensorflow

To take advantage of GPU for better faster training of Neural networks, the system must be equipped with a CUDA enabled GPU card with a compatibility for CUDA 3.5 or higher.

##### Software requirements

- NVIDIA GPU drivers - CUDA 10.1 requires 418.x or higher.
- CUDA Toolkit - TensorFlow supports CUDA 10.1 (TensorFlow >= 2.1.0)
- CUPTI ships with the CUDA Toolkit.
- cuDNN SDK (>= 7.6)

# Scalability Tests

In this section we check how the execution time increases when the data size increases for the two algorithms TGAN and CTGAN.

#### Conclusion
1. For CTGAN algorithm, training time is affected by both the number of rows and columns.
2. For TGAN algorithm, training time is mainly affected by the number of columns.
3. The CTGAN algorithm takes much lower time to train than the TGAN algorithm.

# Statistical Similarity

#### Goal
To calculate the similarity between two tables, our methodology transfers the problem into calculating how different the synthetic data generated by GAN algorithm is from the original data.

#### Methodology

##### 1. Column-wise Similarity Evaluation
- Kullback-Leibler Divergence (KL-divergence)
- Cosine Similarity

##### 2. Table-wise Similarity Evaluation
- Autoencoder for dimension reduction
- PCA and t-SNE for visualization
- Clustering Metric (K-means)

# Privacy Risk Module

#### Goal
Privacy Risk is the most important problem that has led to productivity issues. We need to understand why we need to quantitatively measure privacy risk.

##### Roadmap of Privacy Risk Measurement
1. Exact Matches
2. Closest Matches
3. Privacy at Risk (PaR) metric
4. PaR using Distance Metrics

##### Privacy at Risk methodology
The PaR Metric works by leveraging the idea of confusion with respect to data records. The higher the confusion, the lesser the chance of a person being re-identified.

- Internal Similarity > External Similarity: Ideal Data Point.
- External Similarity > Internal Similarity: Risky Data Point.

# Model Compatibility

#### Goal
Evaluate if models generated using synthetic data are compatible with original data.

#### Methodology
1. One hot encoding
2. Split data into train and test
3. Generate Synthetic Data
4. Standardize variables
5. Model building (Random Forest, XGBoost, KNN, Neural Networks)
6. Hyperparameter tuning
7. Prediction and Metric comparison

#### Final Results Summary
The difference between AUC-ROC metric for synthetic data for a specific model lies within 9% of original data. Any models generated using synthetic data will perform comparatively on original data.

# Ideas for Further Research and Improvements

## Generating Larger Datasets from Small Samples
One can take advantage of statistical similarity to generate datasets that are larger than the original datasets for applications such as training a Neural network.

## Primary Keys
The current implementation for TGAN and CTGAN does not account for the uniqueness of the datasets. ID fields should be preserved.

## Privacy Tuning
1. Post-hoc Analysis of PaR vs Utility.
2. Training GANs by including PaR in the objective function.

## Model Efficiency
GANs are very time-complex. Future research can focus on how best the model building can be optimized.

# Sources
1. https://arxiv.org/pdf/1802.06739.pdf
2. https://blog.cryptographyengineering.com/2016/06/15/what-is-differential-privacy/
3. https://medium.com/georgian-impact-blog/a-brief-introduction-to-differential-privacy-eacf8722283b
4. https://www.techfunnel.com/information-technology/3-types-of-data-anonymization-techniques-and-tools-to-consider/
5. https://towardsdatascience.com/synthetic-data-generation-a-must-have-skill-for-new-data-scientists-915896c0c1ae
6. https://www.ijstr.org/final-print/mar2017/A-Review-Of-Synthetic-Data-Generation-Methods-For-Privacy-Preserving-Data-Publishing.pdf

---

# About the Maintainer

This project is currently maintained by **Shashank Adepu**, an AI Engineer and Data Engineer with over 5 years of experience in designing scalable AI, machine learning, and cloud data solutions. Shashank specializes in building intelligent analytics systems and optimizing large-scale data processing pipelines across healthcare and financial services domains.

**Contact Information:**
- **Email:** adepushashank85@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/shashank1ad/
- **Expertise:** Python, SQL, PySpark, Apache Spark, LangChain, FastAPI, Azure AI, MLOps.