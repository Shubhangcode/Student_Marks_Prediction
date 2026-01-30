import pandas as pd
import numpy as np


df = pd.read_csv("data/processed/processed_data.csv")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
target = "exam_score"
df_model = df[features + [target]].copy()
df_model
le = LabelEncoder()

for col in features:
    df_model[col] = le.fit_transform(df_model[col])
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )
len(y_test)
len(y_train)
models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(),
        "params":{"max_depth": [3,5,10], "min_samples_split": [2,5]}
    },
    "RandomForest":{
        "model":  RandomForestRegressor(),
        "params":{"n_estimators": [50, 100], "max_depth": [5,10]}
    }
}
best_models = []
for name, config in models.items():
  print(f"Training {name}...")
  grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring="neg_mean_squared_error")
  grid_search.fit(X_train, y_train)

  y_pred = grid_search.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)

  best_models.append({
      "model_name": name,
      "best_params": grid_search.best_params_,
      "best_score": grid_search.best_score_,
      "rmse":rmse,
      "r2":r2

  })
  results_df = pd.DataFrame(best_models)
  results_df.sort_values(by="rmse")
  import joblib
best_row = results_df.sort_values(by="rmse").iloc[0]
best_row
results_df = pd.DataFrame(best_models)
best_model_name = best_row["model_name"]
best_model_name
best_model_config = models[best_model_name]
best_model_name
final_model = best_model_config["model"]
final_model.fit(X,y)
joblib.dump(final_model, "best_model.pkl")


