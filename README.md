
# CB9 – A Machine Learning Framework for Forest Fire Prediction in the Nallamala Forest Using NDVI and Synthetic Weather Data

## Team Info
- 22471A05F5 — **GAIRUBOINA NAVEEN KUMAR** ( [LinkedIn](https://www.linkedin.com/in/naveen-kumar-gairuboina-754649276) )
_Work Done: Problem formulation, data collection, preprocessing, model development, result analysis, documentation.

- 22471A05F2 — **DOGHIPARTI VENKATA SAI GIRISH** ( [LinkedIn](https://www.linkedin.com/in/dogiparthi-girish-b0a4bb259) )
_Work Done:  Exploratory data analysis (EDA), feature engineering, visualization, model evaluation.

- 22471A05I6 — **SANIKOMMU NIRUPAM REDDY** ( [LinkedIn](https://www.linkedin.com/in/nirupamreddy-sanikommu-b73995282/) )
_Work Done: Literature survey, dataset validation, performance comparison, report preparation.

---

## Abstract
This project offers a machine learning-based methodology for early warning and prediction of forest fires in India’s ecologically rich Nallamala Forest region. Employing a
combination of satellite-based Normalized Difference Vegetation Index (NDVI) data and synthetically generated weather data from 2012 to 2025, the research constructs a strong model to classify fire events. The pipeline combines MODIS HDF-format NDVI time series with historical temperature and humidity patterns, supplemented by engineered lag features. Ground truth fire events are obtained from MODIS and VIIRS fire archive data sets. For class imbalance in fire event data, the Synthetic Minority Over-sampling Technique (SMOTE) is used. The ultimate predictive model utilizes an ensemble of XGBoost and LightGBM classifiers within a voting approach, with strong potential for operational deployment in forest fire alert systems. This work emphasizes the need for a combination of remote sensing and ML methods for proactive forest management and
climate resilience.

---

## Paper Reference (Inspiration)
👉 **[Projecting Forest Fire Probability in South Korea Under Climate Change, Population, and Forest Management Scenarios Using AI & Process-Based Hybrid Model (FLAM-Net)
  – Hyun-Woo Jo, Myoungsoo Won, Florian Kraxner, Seong Woo Jeon, Yowhan Son,Andrey Krasovskiy,and Woo-Kyun Lee
 ](https://ieeexplore.ieee.org/document/10979267)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Developed a region-specific model for the Nallamala Forest
Integrated long-term NDVI time series with synthetic weather data
Applied SMOTE to handle severe class imbalance
Improved early fire detection capability using ensemble learning
Focused on interpretability and operational feasibility

---

## About the Project
Give a simple explanation of:
- What your project does
  Predicts the likelihood of forest fire occurrence before ignition
- Why it is useful
  Enables early warning, resource planning, and damage reduction
- General project workflow (input → processing → model → output)
  Satellite & weather data → preprocessing & feature engineering → ensemble ML model → fire / non-fire prediction

---

## Dataset Used
👉 **[MODIS NDVI, NASA POWER Weather Data, MODIS & VIIRS Fire Archives](https://drive.google.com/drive/folders/1SLwdLpQZJAQgsIbkG5oeggeErkMsapX8)**

**Dataset Details:**
NDVI: MODIS (250m resolution)
Weather: Temperature, humidity, solar radiation (NASA POWER)
Fire labels: MODIS & VIIRS fire pixels
Time span: 2012–2025
Format: CSV (processed)

---

## Dependencies Used
Python, NumPy, Pandas, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn, Imbalanced-learn (SMOTE)

---

## EDA & Preprocessing
Seasonal fire trend analysis
Missing value handling and normalization
Lag feature creation for NDVI and weather variables
SMOTE applied to balance fire and non-fire classes

---

## Model Training Info
Models used: XGBoost, LightGBM
Ensemble method: Soft voting
Train-test split with stratification
Hyperparameter tuning for optimal performance

---

## Model Testing / Evaluation
Metrics: Accuracy, Precision, Recall, F1-score
ROC–AUC analysis
Confusion matrix for error interpretation

---

## Results
ROC–AUC score: 0.87
Strong discrimination between fire and non-fire events
Seasonal patterns validated climatic influence
Model suitable for early fire warning systems

---

## Limitations & Future Work
Dependency on satellite-derived data
No real-time human activity or ground sensor data
Future work includes real-time deployment, terrain features, and deep learning models (LSTM/Transformers)

---

## Deployment Info
Can be integrated with real-time NDVI and weather APIs
Suitable for forest department monitoring dashboards
Supports early warning and decision-making systems

---
