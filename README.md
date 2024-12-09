# Term Deposit Subscription Prediction Project

This project aims to analyze customer data and predict the likelihood of term deposit subscriptions. Leveraging machine learning techniques, the project provides actionable insights to optimize marketing strategies and enhance campaign effectiveness.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
4. [Modeling Approach](#modeling-approach)
5. [Performance Metrics](#performance-metrics)
6. [Key Insights and Recommendations](#key-insights-and-recommendations)
7. [Technologies Used](#technologies-used)
8. [Acknowledgments](#acknowledgments)

---

## Project Overview
The objective of this project is to develop a predictive model that identifies potential clients likely to subscribe to a term deposit. This involves:
- Cleaning and preparing data for analysis.
- Conducting exploratory data analysis (EDA) to extract key insights.
- Implementing machine learning models to improve prediction accuracy.

---

## Dataset Description
The dataset includes customer demographics, contact details, and campaign interactions. Key features include:
- **Age**, **Job**, **Education**, **Marital Status**
- **Previous Campaign Outcomes** (`poutcome`)
- **Last Contact Duration** (`duration`)
- **Contact Count** (`previous`)
- **Balance**, **Housing Loan**, **Personal Loan**

---

## Exploratory Data Analysis
**Key Observations:**
1. Clients aged 30-45 are the most frequent customers.
2. Longer contact durations correlate with higher subscription rates.
3. Clients with fewer than 2 contact attempts show a higher likelihood of subscribing.
4. Campaign effectiveness diminishes after 3-5 contact attempts.
5. The highest subscription conversion rates occur in March, September, October, and December.

**Insights from Histograms:**
- Most clients work in management or technician roles and are predominantly married.
- Customers with secondary and tertiary education levels show higher conversion rates.

**Correlation Heatmap Highlights:**
- **Duration vs. Subscription**: Strong positive correlation (0.39).
- **Previous Campaign Outcome vs. Contact Timing**: Negative correlation (-0.86), emphasizing the importance of timely follow-ups with successful clients.

---

## Modeling Approach
Multiple machine learning models were implemented and evaluated, of which the best performing models include:
- **Support Vector Classifier (SVC)**
- **Random Forest**
- **XGBoost**
- **Logistic Regression**

### Techniques:
- **Data Balancing**: Applied SMOTE oversampler and random undersampling for balanced class distribution of which random undersampler return a more desirable result.
- **Scalars, Transformation & Encoder**: OneHotEncoder, Robust Scalar and Yeo-Johnson transformation was used to normalize skewed data and transform data.
- **Hyperparameter Tuning**: Optimized models for better accuracy and generalizability.

---

## Performance Metrics
- **Random Forest**: Achieved True Positive Rate (TPR) of 93% with an AUC of 0.92.
- **SVC**: TPR of 89% with an AUC of 0.92.
- **XGBoost**: Highest AUC of 0.93.

---

## Key Insights and Recommendations
1. Focus marketing efforts during high-conversion months (March, September, October, and December).
2. Target customers with long call durations (~500 seconds) and fewer contact attempts (max 2).
3. Prioritize first-time campaign contacts as they represent an untapped customer base.
4. Develop tailored strategies for high-conversion segments like retired individuals and students with tertiary education.

---

## Technologies Used
- **Python**: Data analysis, visualization, and model development.
- **Libraries**: pandas, scikit-learn, XGBoost, TA-Lib, matplotlib, seaborn.

---

