# MovieRater
This project builds a predictive model to estimate IMDb movie ratings based on various movie attributes like genre, director, year, actors, duration, and similar movies. Using a cleaned and feature-engineered dataset, the model leverages XGBoost Regressor to deliver accurate rating predictions.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<b>Project Overview</b></br>
   ----------------
<i>The goal is to train a machine learning model that can predict the IMDb rating of a movie by analyzing trends and patterns across several attributes. The dataset consists of Indian movies extracted from a CSV file (IMDb Movies India.csv). The project focuses on cleaning the data, engineering meaningful features, tuning the model for best performance, and finally saving the model for future use.</i>

<b>1️⃣-Workflow & Pipeline</b></br>
-------------
<i>The project begins by loading and cleaning the data. Key cleaning steps include removing commas from the Votes column, extracting numeric values from Year and Duration, dropping rows with missing essential fields, and converting the Rating column to a numeric format.

Following data cleaning, feature engineering is performed to enhance the model’s predictive power. Several new features are introduced, such as Genre_Success_Rate, Director_Success_Rate, Year_Success_Rate, Actor_Success_Rate, Duration_Success_Rate, and Similar_Avg_Rating. These features capture historical success patterns for genres, directors, actors, release years, and movie durations.

Missing values in engineered columns are imputed using the mean of each respective column, ensuring no data loss during model training. Categorical variables, specifically Genre and Duration_Range, are encoded using one-hot encoding to prepare the dataset for machine learning algorithms.

The dataset is then split into training (80%) and testing (20%) sets, and the input features are standardized using a StandardScaler to ensure uniformity across features, which helps in faster convergence during model training.</i>

<b>2️⃣-Model Building</b></br>
-------------
<i>The model used is XGBoost Regressor, a highly efficient and powerful algorithm well-suited for structured data and tabular datasets. XGBoost is chosen because of its high predictive power, ability to handle missing data internally, built-in regularization techniques that help reduce overfitting, and its support for parallelized training which significantly speeds up the process.

Hyperparameter tuning is performed using RandomizedSearchCV, where multiple combinations of parameters like learning rate, max depth, subsample ratios, and regularization coefficients are explored to find the best-performing model. Cross-validation with 5 folds ensures that the model generalizes well to unseen data.</i>

 <b>3️⃣-Evaluation Metrics</b></br>
 ------------

<i><u>To measure the model's performance, three key metrics are used</u>:</br>

<b>Root Mean Squared Error (RMSE):</b></br>
It represents the square root of the average squared differences between actual and predicted ratings. A lower RMSE indicates that the model’s predictions are closer to the true ratings.

<b>R² Score</u>:</b></br>
Also known as the coefficient of determination, it measures the proportion of the variance in the dependent variable that is predictable from the independent variables. An R² score closer to 1 implies better performance.

<b>Custom Accuracy (±0.5 Tolerance):</b></br>
In this project, a prediction is considered accurate if it falls within ±0.5 of the actual IMDb rating. This custom metric is particularly useful for understanding how "close" the predictions are, even if not perfectly exact.

The custom accuracy is calculated using the following function:</i></br>
```python
def acc_within_half(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) <= 0.5)
```
<img src="github.com/Unique-0743/MovieRater/main/results.png" alt="Local Image" width="300"/>

