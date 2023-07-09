# 📈 Customer Churn Prediction with LightGBM, CatBoost, TabTransformer, and AutoGluon-Tabular in Amazon SageMaker 🚀

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/vivek7208/Customer-Churn-Prediction/blob/main/notebook.ipynb)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Preview in nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/vivek7208/Customer-Churn-Prediction/blob/main/notebook.ipynb)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vivek7208/Customer-Churn-Prediction/blob/main/notebook.ipynb)

This repository contains a Jupyter notebook 📓 that demonstrates how to use machine learning (ML) 🧠 for the automated identification of unhappy customers 😞, also known as customer churn prediction. The notebook employs Amazon SageMaker's implementation of [LightGBM](https://lightgbm.readthedocs.io/en/latest/) 💡, [CatBoost](https://catboost.ai/) 🐱, [TabTransformer](https://arxiv.org/abs/2012.06678) 🔄, and [AutoGluon-Tabular](https://auto.gluon.ai/stable/index.html) 📊 algorithms to train and host a customer churn prediction model using Amazon SageMaker's Automatic Model Tuning (AMT) 🎛️.

## 📝Overview

Losing customers is costly for any business. 💸 Identifying unhappy customers early gives businesses a chance to offer incentives to stay. However, ML models rarely give perfect predictions. This notebook showcases how to incorporate the relative costs of prediction mistakes when determining the financial outcome of using ML.

The ML models used in this notebook are trained in two scenarios:

1. Training a tabular model on the customer churn dataset with AMT. 📊
2. Using the trained tabular model to perform inference, i.e., classifying new samples. 🧮

In the end, the performance of the four models trained with AMT is compared on the same test data. 📈

## 📘Notebook Contents

The notebook contains the following sections:

1. **Set Up:** This part involves setting up the notebook and importing necessary libraries. 📚

2. **Data Preparation and Visualization:** This section deals with preparing the dataset for the machine learning models and visualizing the data for a better understanding of it. 📉 The data visualization includes histograms for each numeric feature. The dataset used is a publicly available set containing 5,000 records with 21 attributes describing the profile of a customer of an unknown US mobile operator. The data is processed and visualized to understand its distribution and its relation with the target variable 'Churn?'. This section also includes converting the target attribute to binary and moving it to the first column of the dataset to meet the requirements of SageMaker's built-in tabular algorithms.

3. **Train A LightGBM Model with AMT:** This part involves training a LightGBM model with AMT and also includes sub-sections on retrieving training artifacts, setting training parameters, starting the training process, deploying and running inference on the trained model, and evaluating the prediction results. 🎯

4. **Train A CatBoost model with AMT:** Similar to the previous section, this part focuses on the CatBoost model. 🐱‍💻

5. **Train A TabTransformer model with AMT:** This section focuses on training a TabTransformer model with AMT. 🌀

6. **Train An AutoGluon-Tabular model:** This section is about training an AutoGluon-Tabular model. 🏎️

7. **Compare Prediction Results of Four Trained Models on the Same Test Data:** In the final section, the performance of the four trained models - LightGBM, CatBoost, TabTransformer, and AutoGluon-Tabular - is compared using the same test data. The comparison includes a plot of the confusion matrix for each model. 📊

## 🤖Models Used

This notebook uses the following machine learning models:

### 🌳LightGBM

[LightGBM](https://lightgbm.readthedocs.io/en/latest/) is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages: faster training speed and higher efficiency, lower memory usage, better accuracy, parallel and GPU learning supported, capable of handling large-scale data.

### 🐈‍⬛CatBoost

[CatBoost](https://catboost.ai/) is an open-source gradient boosting on decision trees library with categorical features support. The CatBoost library can be used to solve both classification and regression tasks. The library provides all the functionality necessary to solve classification, regression and ranking tasks, supports GPU acceleration and handling categorical features.

### 📊TabTransformer

[TabTransformer](https://arxiv.org/abs/2012.06678) is a model introduced in the paper "TabTransformer: Tabular Data Modeling Using Contextual Embeddings". It treats each feature as a token and applies a transformer model to capture the interactions between features. It's especially powerful for tabular data with high cardinality categorical features.

### 🚀AutoGluon-Tabular

[AutoGluon-Tabular](https://auto.gluon.ai/stable/index.html) is an automated machine learning model designed for tabular data. It automatically prepares the data, selects the necessary features, and chooses the best model and hyperparameters to solve the problem. AutoGluon-Tabular can save a lot of time and effort with great performance.

## 🖼️Images

![image](https://github.com/vivek7208/Customer-Churn-Prediction/assets/65945306/f95c5cfa-13ef-4c4d-a2b1-b7230aefe97a)



|                         | LightGBM with AMT | CatBoost with AMT | TabTransformer with AMT | AutoGluon-Tabular |
|-------------------------|------------------:|------------------:|------------------------:|------------------:|
| Accuracy                |          0.895556 |          0.953333 |                0.960000 |          0.980000 |
| F1                      |          0.894855 |          0.953229 |                0.960526 |          0.980044 |
| AUC                     |          0.965412 |          0.991407 |                0.992040 |          0.997926 |

Each column represents a different model, and the rows represent different metrics (Accuracy, F1, and AUC) for each model. 📝

## 🧹Cleaning Up

To avoid incurring unnecessary charges, remember to delete the endpoint corresponding to the trained model. Instructions on how to do this are provided at the end of the notebook. 🧾

## 🔧Usage

You can use this notebook to evaluate the performance of LightGBM, CatBoost, TabTransformer, and AutoGluon-Tabular on your own dataset. The notebook was tested in Amazon SageMaker Studio on ml.t3.medium instance with Python 3 (Data Science) kernel. 🐍
