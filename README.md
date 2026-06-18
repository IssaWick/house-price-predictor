# 🏠 House Price Predictor

A machine learning web app that predicts house prices from features like area, number of rooms, and amenities. Built with Python, scikit-learn, and Streamlit, and deployed to the web.

### 🔗 Live Demo
**[https://house-price-predictor-dmzrwfw6uahrn8f2dbjieb.streamlit.app/](https://house-price-predictor-dmzrwfw6uahrn8f2dbjieb.streamlit.app/)**

---

## Overview

This project walks through a complete, real-world machine learning workflow: exploring a dataset, training and comparing models, building an interactive dashboard, and deploying it as a public web app. A user enters the details of a house and instantly receives an estimated price.

## Features

- Interactive form for entering house details (area, bedrooms, bathrooms, amenities, etc.)
- Live price prediction powered by a trained regression model
- Built-in data explorer to view the dataset and the price-vs-area relationship
- Deployed and publicly accessible

## Tech Stack

- **Python 3.11**
- **scikit-learn** — model training
- **pandas / numpy** — data handling
- **Streamlit** — dashboard and web app
- **joblib** — saving and loading the model
- **Anaconda** — environment management
- **Git & GitHub** — version control
- **Streamlit Community Cloud** — deployment

## Dataset

The [Housing Prices Dataset](https://www.kaggle.com/datasets) (~545 records). The target variable is `price`, predicted from features including `area`, `bedrooms`, `bathrooms`, `stories`, `parking`, and yes/no amenities such as `mainroad`, `basement`, `airconditioning`, `prefarea`, and `furnishingstatus`.

## Model & Results

Two models were trained and compared on a held-out test set:

| Model | R² Score |
|---|---|
| **Linear Regression** ✅ | **0.653** |
| Random Forest | 0.612 |

Linear Regression was selected as the final model. On the test set it achieved an **R² of 0.653** (explaining ~65% of price variation), a **MAE of ~970,043**, and an **RMSE of ~1,324,507**. A key takeaway: the simpler model outperformed the more complex one — a reminder to always compare rather than assume.

## Project Structure

```
house-price-predictor/
├── data/
│   └── Housing.csv          # dataset
├── notebooks/
│   ├── 01_eda.ipynb         # exploratory data analysis
│   └── 02_modeling.ipynb    # preprocessing, training, evaluation
├── models/
│   ├── house_price_model.pkl    # trained model
│   └── model_columns.pkl        # expected feature columns
├── app.py                   # Streamlit dashboard
├── requirements.txt         # dependencies
├── .gitignore
└── README.md
```

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/IssaWick/house-price-predictor.git
cd house-price-predictor

# 2. Create and activate the environment
conda create -n streamlit-ml python=3.11 -y
conda activate streamlit-ml

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## Possible Improvements

- Show a confidence range alongside each prediction
- Add more visualizations to the data explorer
- Try additional models or feature engineering to improve accuracy

---

*Built as an end-to-end machine learning project covering data exploration, modeling, dashboard development, and deployment.*
