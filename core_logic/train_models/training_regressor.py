import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR     it is too slow and can't outperform ensemble models so dropping it
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

df=pd.read_parquet("data/model_data.parquet")
print(df.head())
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
df["future_volatility_20"] = (df.groupby("ticker")["future_return"]
      .transform(lambda x: x.shift(-1).rolling(20).std()))
df = df.dropna(subset=["future_volatility_20"]).reset_index(drop=True)
print(df["future_volatility_20"][:5])

X = df.drop(columns=["future_return","ticker","date","future_volatility_20"])
y = df["future_volatility_20"]
split = int(len(df) * 0.8)

x_train = X.iloc[:split]
x_test  = X.iloc[split:]

y_train = y.iloc[:split]
y_test  = y.iloc[split:]

tscv = TimeSeriesSplit(n_splits=5)

search_spaces = {

    "Ridge": {
        "model": Ridge(random_state=42),
        "params": {
            "alpha": [0.01, 0.1, 1, 10, 100],
        },
        "n_iter": 5,
        "needs_scaling": True,
    },

    "RF": {
        "model": RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": [200,300,500,800],
            "max_depth": [None,4,6,8,12,16],
            "min_samples_split": [2,5,10,20],
            "min_samples_leaf": [1,2,4,8],
            "max_features": ["sqrt","log2",0.5,0.8],
        },
        "n_iter": 25,
        "needs_scaling": False,
    },

    "LGBM": {
        "model": LGBMRegressor(
            random_state=42,
            verbose=-1
        ),
        "params": {
            "n_estimators": [200,300,500,800],
            "num_leaves": [15,31,63,127],
            "max_depth": [-1,4,6,8],
            "learning_rate": [0.01,0.03,0.05,0.1],
            "subsample": [0.6,0.8,1.0],
            "colsample_bytree": [0.6,0.8,1.0],
            "min_child_samples": [10,20,30,50],
        },
        "n_iter": 15,
        "needs_scaling": False,
    },

    "CatBoost": {
        "model": CatBoostRegressor(
            random_seed=42,
            verbose=0
        ),
        "params": {
            "iterations": [300,500,800],
            "depth": [4,6,8,10],
            "learning_rate": [0.01,0.03,0.05,0.1],
            "l2_leaf_reg": [1,3,5,7,9],
        },
        "n_iter": 15,
        "needs_scaling": False,
    },

    "XGB": {
        "model": XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            eval_metric="rmse"
        ),
        "params": {
            "n_estimators": [200,300,500,800],
            "max_depth": [3,4,6,8],
            "learning_rate": [0.01,0.03,0.05,0.1],
            "subsample": [0.6,0.8,1.0],
            "colsample_bytree": [0.6,0.8,1.0],
            "min_child_weight": [1,3,5,10],
        },
        "n_iter": 15,
        "needs_scaling": False,
    },

}
scaler = StandardScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns,index=x_train.index)
x_test_scaled = pd.DataFrame(scaler.transform(x_test),columns=x_test.columns,index=x_test.index)

best_score = float("inf")
best_model = None
best_model_name = None
best_params = None
results = {}
for name, cfg in search_spaces.items():
    xt = x_train_scaled if cfg["needs_scaling"] else x_train
    xte = x_test_scaled if cfg["needs_scaling"] else x_test

    search = RandomizedSearchCV(
        estimator=cfg["model"],
        param_distributions=cfg["params"],
        n_iter=cfg["n_iter"],
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    print("\n")
    print(name,"  training........................")
    search.fit(xt, y_train)

    tuned_model = search.best_estimator_
    
    y_pred = tuned_model.predict(xte)
    absolute_error_score=mean_absolute_error(y_test,y_pred)
    mean_squared_score=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(
        f"{name}"
        f"\nCV MAE: {-search.best_score_:.6f}"  # - because its value is returned negative by random search cv
        f"\nTest MAE: {absolute_error_score:.6f}"
        f"\nRMSE: {rmse:.6f}"
        f"\nR²: {r2:.4f}"
    )
    print(search.best_params_)

    results[name] = {
    "cv_mae": -search.best_score_,
    "test_mae": absolute_error_score,
    "rmse": rmse,
    "r2": r2,
    "params": search.best_params_
    }
    
    if absolute_error_score < best_score:
        best_score = absolute_error_score
        best_model = tuned_model
        best_model_name = name
        best_params = search.best_params_
    
for name, r in results.items():
    print(f"{name:10s} CV: {r['cv_mae']:.4f}  Holdout: {r['test_mae']:.4f}")

print("=" * 60)
print("Best Model :", best_model_name)
print("Best MAE   :", best_score)
print("Best Params:", best_params)

if hasattr(best_model, "feature_importances_"):
    importance = pd.DataFrame({"Feature": X.columns, "Importance": best_model.feature_importances_}).sort_values("Importance", ascending=False)
    print(importance.head(20))

joblib.dump(best_model, "models/best_final_regression_model.pkl")
joblib.dump(scaler, "models/regression_scaler.pkl")
joblib.dump(
    list(X.columns),
    "models/regression_feature_columns.pkl"
)
results_df = pd.DataFrame(results).T
results_df.to_csv("models/regression_results.csv", index=True)
print("Everything saved successfully ...............finally after many efforts........")