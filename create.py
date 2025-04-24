import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
import joblib
import dataprocessing


df=dataprocessing.df

# features extraction from the processed dataset for  analysis and prediction purpose.

features = ["Votes","Year","Duration",
            "Genre_Success_Rate","Director_Success_Rate",
            "Year_Success_Rate","Actor_Success_Rate",
            "Duration_Success_Rate","Similar_Avg_Rating"] + list(dataprocessing.cat_df.columns)
df.dropna(subset=["Rating"], inplace=True)
# now we segregate inputs and outputs data
X = df[features]
y = df["Rating"]

# splitting the available data for training and testing (80% training and 20% testing)
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# through scaling we set or transform the values into a small range difference
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test  = scaler.transform(X_test_df)
joblib.dump(scaler, "scaler.pkl")

# accuracy calculator(with 0.5 value tolerance)
def acc_within_half(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) <= 0.5)
acc_scorer = make_scorer(acc_within_half, greater_is_better=True)

# we declared the hyperparameters of xgboost regressor
param_dist = {
    'n_estimators': [500, 700],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.05, 0.07],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.5],
    'reg_lambda': [1, 1.5],
    'min_child_weight': [1, 3, 5]
}
# created xgbregressor
xgb = XGBRegressor(random_state=42, verbosity=0)
#perform randomized search to predict the best hyper parameter through which we train the training set.
search = RandomizedSearchCV(
    xgb, param_distributions=param_dist, n_iter=40,
    scoring=acc_scorer, cv=5,
    random_state=42, n_jobs=-1, verbose=1
)
# training starts.
search.fit(X_train, y_train)
best_xgb = search.best_estimator_

# evaluation by calculating rms,r2_scord and accuracy.
train_metrics = {
    'RMSE': root_mean_squared_error(y_train, best_xgb.predict(X_train)),
    'R2':   r2_score(y_train, best_xgb.predict(X_train)),
    'Accuracy ±0.5 (%)': acc_within_half(y_train, best_xgb.predict(X_train))*100
}
test_metrics = {
    'RMSE': root_mean_squared_error(y_test, best_xgb.predict(X_test)),
    'R2':   r2_score(y_test, best_xgb.predict(X_test)),
    'Accuracy ±0.5 (%)': acc_within_half(y_test, best_xgb.predict(X_test))*100
}

print("Best XGBoost Hyperparameters:")
print(search.best_params_)
print("Performance:")
print(pd.DataFrame([train_metrics, test_metrics], index=["Train","Test"]))

# saving model.pkl file.
joblib.dump(best_xgb, r"D:\datascience\model.pkl")
print("model.pkl file saved")
