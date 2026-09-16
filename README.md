# Employee Attrition Prediction & Feature Analysis

A machine learning web application that predicts whether an employee is likely
to leave a company, and explains *why* using feature importance analysis —
built end-to-end with a trained Random Forest model served through a Flask web app.

# how our UI looks like and works 
   ![Form](ss1.png)
   ![Result](ss2.png)


## What it does

- Takes employee details (age, income, job satisfaction, overtime, tenure, etc.)
  through a web form
- Predicts attrition risk using a tuned Random Forest classifier
- Displays a confidence score and the top factors driving that specific prediction
- Shows global feature importance across the whole model

## Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (Random Forest, GridSearchCV), pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML, CSS

## Model Performance

Trained and evaluated on the IBM HR Analytics Employee Attrition dataset (1,470 records).

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| Logistic Regression (baseline) | 0.69 | 0.38 | 0.49 |
| Random Forest (GridSearchCV-tuned) | 0.46 | **0.53** | 0.50 |

Random Forest was selected over the baseline despite a near-identical F1 score,
because **recall matters more for this problem**: failing to flag an employee
who actually leaves (a false negative) is costlier to a business than one extra
false alarm. Hyperparameters (`n_estimators=300`, `max_depth=8`,
`min_samples_leaf=4`, `class_weight='balanced'`) were selected via 5-fold
cross-validated grid search optimizing for F1.

## Top predictive factors

Monthly Income, OverTime, Age, Total Working Years, Years at Company, and
Years With Current Manager were the strongest predictors of attrition,
identified via the model's built-in feature importance scores.

## Running it locally

\`\`\`bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-prediction.git
cd employee-attrition-prediction
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py
\`\`\`
Then open `http://127.0.0.1:5000` in your browser.

*(To retrain the model from scratch instead of using the included saved model,
run `python train_model.py` first.)*

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle)
