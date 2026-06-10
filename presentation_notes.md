# Presentation Notes — California Housing Value Lab

## 1. Hook / business case
Our app helps a real-estate analyst understand which neighborhood characteristics are linked to higher California median home values and estimate value for a new area.

## 2. Data overview
- Rows: 20,640
- Input features: 8
- Target: `MedianHouseValue_100k`
- Target type: continuous, which fits the project requirement to use linear regression.
- Missing values: none.

## 3. Visual story
- Start with the target distribution to show the range of values.
- Use the scatter plot to show that median income has a strong relationship with median home value.
- Use the map to show location patterns: coastal and metro areas generally have higher values.
- Use the correlation matrix to identify strongest relationships and possible noisy variables.

## 4. Model story
- The model is Linear Regression from scikit-learn.
- R-squared explains how much of the target variation is explained by the selected features.
- MAE and RMSE show prediction error in dollars.
- The actual-vs-predicted plot shows where the model is accurate and where it misses.
- The simulator lets the audience change feature values and get a predicted median home value.

## 5. Improvements
- Add richer local economic data, school ratings, commute time, and current market conditions.
- Remove noisy variables or add more useful ones to improve R-squared and reduce prediction error.
- Later, compare nonlinear models, but keep linear regression for the midterm requirement.
