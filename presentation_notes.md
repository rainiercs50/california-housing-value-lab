# Presentation Notes — California Housing Value Lab

Use these notes as your private speaking guide. They are not displayed inside the app.

## 1. Opening

Today I am presenting the California Housing Value Lab. The goal is to understand what neighborhood characteristics are connected to higher median home values, and then use those patterns to make a transparent value prediction.

## 2. Dataset overview

The dataset contains 20,640 California communities. It includes income, house age, rooms, bedrooms, population, occupancy, latitude, longitude, engineered ratios, and categorical segments such as income band, coastal category, age group, and region. The target is median house value measured in 100,000-dollar units.

## 3. Visual findings

Start with the target distribution. The values are not perfectly normal, and there is a visible high-value cap.

Then show the scatter explorer. Median income has the clearest positive relationship with home value. Higher-income communities generally have higher median values.

Next, show the categorical bar chart. The categories make the story easier to explain because high-income and coastal/region groups tend to show higher average values.

Then show the map. Geography matters because expensive areas cluster near certain coastal and metro regions.

Finally, show the correlation tab. This supports why income and location belong in the model.

## 4. Model explanation

The model is Linear Regression. Numeric variables are scaled and categorical variables are one-hot encoded. The main metric is R², which means the percentage of variation in home values explained by the selected features. MAE and RMSE show the average size of prediction errors in dollars.

Do not say the model is perfect. Say it is useful because it is interpretable and connects the visuals to a prediction.

## 5. Simulator demo

Use the simulator to change a realistic input such as median income or region. Explain that the prediction updates because the same trained model is being applied to the hypothetical neighborhood profile.

## 6. Conclusion

The strongest pattern is that income is closely connected to home value, but geography and household structure also matter. Linear Regression gives a clear first prediction model. The main limitations are that the data is historical, high values are capped, and housing markets are more complex than a linear model can fully capture.
