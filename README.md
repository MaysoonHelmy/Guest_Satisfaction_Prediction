# Guest Satisfaction Prediction

This project aims to predict guest satisfaction scores based on Airbnb listings and host data using advanced data preprocessing, feature engineering, and machine learning models. The end goal is to help hosts improve their services and optimize their listings based on factors that most impact guest reviews.

## 📁 Project Team

- Maysoon Helmy
- Roaa Mohamed Lotfy
- Menna Yasser Hamed
- Maya Samah Ahmed
- Arwa Abu Zlema

## 📌 Project Overview

This project covers:
- Data cleaning and preprocessing
- Feature engineering and selection
- Regression and classification modeling
- Evaluation and interpretation of model performance

## 🛠️ Key Steps & Techniques

### 🧹 Data Cleaning
- Removed non-informative columns (e.g., `ID`, `thumbnail_url`, `square_feet`)
- Handled missing values using techniques like mode imputation, contextual defaulting, and group-based filling.

### 🧠 Feature Engineering
- Custom features like:
  - `host_response_power`
  - `bedroom_quality`
  - `price_value`
  - `review_consistency`
- Transformation of text features using TF-IDF
- Aggregation and interaction features
- Seasonality analysis via `seasonal_demand`

### 🔄 Encoding
- Binary encoding for true/false columns
- Label encoding for nominal categories
- Ordinal encoding for ranked data (e.g., host response time)
- Deep learning embeddings for `host_neighbourhood`

### 📊 Feature Selection
- Correlation analysis
- ANOVA and Mutual Information (SelectKBest)
- Chi-square tests for categorical classification

### 📈 Modeling

#### Regression
- Linear Regression
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost *(Best model with R² ≈ 0.89)*
- Random Forest

#### Classification
- Logistic Regression
- SVM
- Random Forest
- Gradient Boosting
- XGBoost *(Best model with best accuracy and speed balance)*

### 🧪 Evaluation
- Cross-validation (5-fold)
- Train-Test Split
- Accuracy, R², and time complexity metrics
- Visualizations: distributions, boxplots, and model performance graphs

## 📊 Results

### Best Models:
- **Regression**: CatBoost
- **Classification**: XGBoost

### Key Findings:
- Tree-based ensemble models outperform simpler models
- Feature engineering significantly boosts model performance
- Proper handling of missing data and outliers is crucial for model stability

## 📂 Folder Structure

```
.
├── data/
├── notebooks/
├── models/
├── results/
├── src/
├── README.md
└── requirements.txt
```

## 📚 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

```bash
# Run the Regression Model App (Milestone 1)
python Milestone 1 GUI/Guest_Rating_Predictor.py

# Run the Classification Model App (Milestone 2)
python Milestone 2 GUI/Guest_Satisfaction_Predictor.py
```

## 📌 Conclusion

This project was developed as part of the Machine Learning course at Ain Shams University, Faculty of Computer and Information Science. It showcases the practical application of machine learning techniques in a real-world domain — the hospitality and rental industry.

Through rigorous data preprocessing, advanced feature engineering, and comprehensive model evaluation, we successfully built models that predict guest satisfaction with high accuracy. Our analysis demonstrated that:

📈 Tree-based ensemble models like CatBoost and XGBoost consistently outperform simpler models in both regression and classification tasks.

🧠 Thoughtful feature engineering — especially domain-aware constructs like host_commitment and review_consistency — significantly enhanced model performance.

🧹 Proper handling of missing values, outliers, and text preprocessing was critical to building a reliable dataset.

📊 Evaluation metrics, such as R² and accuracy, combined with visual diagnostics, allowed us to fine-tune and validate model robustness.

In summary, this project provided a strong foundation in applying machine learning techniques end-to-end — from raw data to deployment-ready insights — and deepened our understanding of how predictive models can inform decision-making in user-centric platforms like Airbnb.
