# Employee Attrition Prediction and Feature Analysis Using Machine Learning

## 1. Folder structure — where everything goes

```
Employee-Attrition-Prediction/            ← your existing project root
├── venv/                                 ← you already have this, leave as-is
├── WA_Fn-UseC_-HR-Employee-Attrition.csv ← you already have this, leave in root
├── requirements.txt                      ← NEW, place in root
├── train_model.py                        ← NEW, place in root
├── app.py                                ← NEW, place in root
├── notebooks/
│   ├── 01_data_exploration.py            ← NEW (reference only, ignore this one)
│   └── 01_data_exploration.ipynb         ← NEW, this is the real notebook — open this in VS Code
├── templates/
│   ├── index.html                        ← NEW, the input form page
│   └── result.html                       ← NEW, the prediction result page
├── static/
│   └── style.css                         ← NEW, all styling
└── model/                                ← EMPTY for now — gets filled automatically
                                              when you run train_model.py
```

Copy every file/folder above (except `venv` and the CSV, which you already have)
into your `Employee-Attrition-Prediction` folder, preserving the same
sub-folder names (`notebooks/`, `templates/`, `static/`, `model/`).

Flask specifically **requires** the `templates/` and `static/` folder names —
don't rename them, or Flask won't find your HTML/CSS.

## 2. Order of operations — run things in this sequence

### Step A — Install any missing packages
With your `(venv)` activated in VS Code's terminal:
```bash
pip install -r requirements.txt
```

### Step B — Explore the data (Day 3-4)
Open `notebooks/01_data_exploration.ipynb` in VS Code, select your venv as the
kernel, and run all cells top to bottom (`Run All`). This is where you visually
inspect the dataset, see the charts, and understand what you're working with.
This notebook also saves `cleaned_attrition_data.csv` — informational only,
`train_model.py` does its own cleaning independently so it can run standalone.

### Step C — Train the model (Day 5-9)
In the terminal, from the project root:
```bash
python train_model.py
```
This will:
- Clean and encode the dataset
- Train a Logistic Regression baseline AND a Random Forest
- Print accuracy/precision/recall/F1 for both, so you can see Random Forest winning
- Save 4 files into `model/`: the trained model, the scaler, the label encoders, and the feature column order
- Save a feature importance chart to `static/feature_importance.png`

**You must run this before starting the Flask app** — `app.py` loads the files this script creates.

### Step D — Run the web app (Day 11-14)
```bash
python app.py
```
Then open the URL it prints (usually `http://127.0.0.1:5000`) in your browser.
Fill in the form, hit **Predict Attrition Risk**, and you'll land on a result
page with a risk gauge and the top 5 factors the model weighed most heavily.

## 3. What each file actually does (for your understanding + viva prep)

| File | Role |
|---|---|
| `01_data_exploration.ipynb` | Understand the raw data: shape, types, missing values, useless columns, correlations |
| `train_model.py` | Turns raw CSV → trained, saved model + supporting encoders |
| `app.py` | Flask server: shows the form, receives input, encodes it identically to training, predicts, shows result |
| `templates/index.html` | The data-entry form, grouped into Personal / Job / Compensation / Satisfaction sections |
| `templates/result.html` | Displays the prediction + feature importance breakdown |
| `static/style.css` | All visual styling (dark theme, gauge, bars) |
| `model/*.pkl` | Saved model artifacts — this is what makes predictions possible without retraining every time |

## 4. Why I made these specific choices (for your report's "methodology" section)

- **Random Forest over Logistic Regression**: trees naturally handle a mix of
  numeric and categorical-encoded features and capture non-linear patterns
  (e.g. "risk spikes specifically when OverTime=Yes AND JobSatisfaction is low"
  — a linear model can't express that combination as easily).
- **`class_weight='balanced'`**: our dataset has far more "stayed" than "left"
  employees. Without this, the model could get high accuracy just by always
  predicting "stayed" — this setting forces it to actually learn the minority
  class too.
- **LabelEncoder saved and reused**: the exact same text-to-number mapping used
  in training must be used on new user input, or predictions would be
  nonsensical (e.g. "Sales" encoded as 2 in training but 0 at prediction time).
- **Feature importance instead of SHAP/LIME**: global feature importance from
  Random Forest is simple to explain in a viva ("the model tracks how much
  each feature reduces prediction error across all its trees") — SHAP is more
  powerful but adds complexity beyond what's needed for a 15-day internship
  project.

## 5. What to explore/tweak next (optional, if time permits)
- Try `GridSearchCV` to tune `n_estimators` / `max_depth` and see if performance improves
- Add a confusion matrix plot to the notebook for your report
- Add input validation messages in the Flask form
