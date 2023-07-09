# 📈 Customer Churn Prediction with LightGBM, CatBoost, TabTransformer, and AutoGluon-Tabular in Amazon SageMaker 🚀

This repository contains a Jupyter notebook 📓 that demonstrates how to use machine learning (ML) 🧠 for the automated identification of unhappy customers 😞, also known as customer churn prediction. The notebook employs Amazon SageMaker's implementation of [LightGBM](https://lightgbm.readthedocs.io/en/latest/) 💡, [CatBoost](https://catboost.ai/) 🐱, [TabTransformer](https://arxiv.org/abs/2012.06678) 🔄, and [AutoGluon-Tabular](https://auto.gluon.ai/stable/index.html) 📊 algorithms to train and host a customer churn prediction model using Amazon SageMaker's Automatic Model Tuning (AMT) 🎛️.

## 🌐 Overview

Losing customers is costly for any business 💼. Identifying unhappy customers early gives businesses a chance to offer incentives to stay 🎁. However, ML models rarely give perfect predictions. This notebook showcases how to incorporate the relative costs of prediction mistakes when determining the financial outcome of using ML 💰.

The ML models used in this notebook are trained in two scenarios:

1️⃣ Training a tabular model on the customer churn dataset with AMT.
2️⃣ Using the trained tabular model to perform inference, i.e., classifying new samples.

In the end, the performance of the four models trained with AMT is compared on the same test data 🧪.

## 📖 Notebook Contents

The notebook contains the following sections:

1️⃣ **Set Up:** This part involves setting up the notebook and importing necessary libraries 📚.

2️⃣ **Data Preparation and Visualization:** This section deals with preparing the dataset for the machine learning models and visualizing the data for a better understanding of it 📊. 

3️⃣ **Train A LightGBM Model with AMT:** This part involves training a LightGBM model with AMT 💡.

4️⃣ **Train A CatBoost model with AMT:** Similar to the previous section, this part focuses on the CatBoost model 🐱.

5️⃣ **Train A TabTransformer model with AMT:** This section focuses on training a TabTransformer model with AMT 🔄.

6️⃣ **Train An AutoGluon-Tabular model:** This section is about training an AutoGluon-Tabular model 📊.

7️⃣ **Compare Prediction Results of Four Trained Models on the Same Test Data:** In the final section, the performance of the four trained models - LightGBM, CatBoost, TabTransformer, and AutoGluon-Tabular - is compared using the same test data 🏁.

## 🤖 Models Used

This notebook uses the following machine learning models:

### 💡 LightGBM

[LightGBM](https://lightgbm.readthedocs.io/en/latest/) is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient.

### 🐱 CatBoost

[CatBoost](https://catboost.ai/) is an open-source gradient boosting on decision trees library with categorical features support.

### 🔄 TabTransformer

[TabTransformer](https://arxiv.org/abs/2012.06678) is a model introduced in the paper "TabTransformer: Tabular Data Modeling Using Contextual Embeddings".

### 📊 AutoGluon-Tabular

[AutoGluon-Tabular](https://auto.gluon.ai/stable/index.html) is an automated machine learning model designed for tabular data.

## 📸 Images

![image](https://github.com/vivek7208/Customer-Churn-Prediction/assets/65945306/f95c5cfa-13ef-4c4d-a2b1-b7230aefe97a)

|                         | LightGBM with AMT | CatBoost with AMT | TabTransformer with AMT | AutoGluon-Tabular |
|-------------------------|------------------:|------------------:|------------------------:|------------------:|
| Accuracy                |          0.895556 |          0.953333 |                0.960000 |          0.980000 |
| F1                      |          0.894855 |          0.953229 |                0.960526 |          0.980044 |
| AUC                     |          0.965412 |          0.991407 |                0.992040 |          0.997926 |

Each column represents a different model, and the rows represent different metrics (Accuracy, F1, and AUC) for each model 📈.

## 🧹 Cleaning Up

To avoid incurring unnecessary charges, remember to delete the endpoint corresponding to the trained model. Instructions on how to do this are provided at the end of the notebook 🧾.

## 🎮 Usage

You can use this notebook to evaluate the performance of LightGBM, CatBoost, TabTransformer, and AutoGluon-Tabular on your own dataset. The notebook was tested in Amazon SageMaker Studio on ml.t3.medium instance with Python 3 (Data Science) kernel 🐍.
