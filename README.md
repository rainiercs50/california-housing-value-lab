# California Housing Value Lab — Streamlit Midterm Project

A polished Streamlit midterm app that analyzes California housing data and predicts median neighborhood house value with Linear Regression.

## Project story

**Problem:** A real-estate analyst wants to understand which neighborhood characteristics are connected to higher California home values and estimate median value for a hypothetical neighborhood.

**Dataset:** `california_housing_midterm.csv`

- 20,640 rows
- 16 variables after transparent feature engineering
- Numerical variables: income, house age, rooms, bedrooms, population, occupancy, latitude, longitude, engineered ratios
- Categorical variables: income band, age group, coastal category, California region
- Continuous target: `MedianHouseValue_100k`, median house value in $100,000 units

## App sections

1. **Introduction + Exploration**
   - Dataset overview
   - Row/variable counts
   - Missing values
   - Data types
   - Summary statistics
   - Categorical overview

2. **Interactive Data Visualization**
   - Target distribution
   - Scatter explorer with dropdowns
   - Bar charts for categorical analysis
   - Line chart by house age
   - Interactive California map
   - Correlation matrix

3. **Linear Regression Prediction**
   - scikit-learn Linear Regression
   - Train/test split controls
   - Numeric scaling + categorical one-hot encoding
   - R², MAE, RMSE
   - Actual vs predicted plot
   - Residual plot
   - Coefficient chart
   - Live prediction simulator

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files for deployment

- `app.py` — Streamlit app entrypoint
- `requirements.txt` — required Python packages
- `california_housing_midterm.csv` — deployment dataset
- `.streamlit/config.toml` — visual theme config
- `presentation_notes.md` — separate speaking notes (not displayed inside the app)
- `design_notes.md` — project design rationale

## Deployment instructions

1. Upload this repository to GitHub.
2. Go to Streamlit Community Cloud: <https://share.streamlit.io/>.
3. Create a new app from the GitHub repository.
4. Set the branch to `main` and the app file to `app.py`.
5. Deploy and submit the public Streamlit app link.

## Model note

The required model is Linear Regression. The app presents the model honestly: R² means variation explained, not perfect accuracy. A moderate R² is acceptable because the model is interpretable and the presentation explains limitations such as historical data and capped high values.
