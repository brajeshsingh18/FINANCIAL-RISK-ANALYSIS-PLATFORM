# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
# import joblib

# df=pd.read_parquet("data/model_data.parquet")
# df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
# x = df.drop(columns=["future_return","ticker","date"])
# y=df["future_return"]

# # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# # we can't use train test split here  because data should be from past to future but this split can make it random

# split = int(len(df) * 0.8)
# x_train = x.iloc[:split]
# x_test = x.iloc[split:]
# y_train = y.iloc[:split]
# y_test = y.iloc[split:]

# print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# models={
#     "RF":RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1),
#     "LGBM":LGBMRegressor(n_estimators=300,random_state=42),
#     "Catboost":CatBoostRegressor(random_seed=42,verbose=0),
#     "XGB":XGBRegressor(random_state=42,n_estimators=300)
# }
# best_score = float("-inf")
# best_model = None
# best_model_name = None
# for model_name,model in models.items():
#     model.fit(x_train,y_train)
#     y_pred=model.predict(x_test)
#     score=r2_score(y_test,y_pred)
#     print("MSE: ",mean_squared_error(y_test,y_pred))
#     print("MAE: ",mean_absolute_error(y_test,y_pred))
#     print("model: ",model_name)
#     print("accuracy: ",score)
#     print("*****************************")
#     if score>best_score:
#         best_score=score
#         best_model=model
#         best_model_name=model_name

# print(best_model," : ",best_model_name , " : ",best_score)
# joblib.dump(
#     best_model,
#     "models/best_model.pkl"
# )

# the regression model training framework we just prepared is not performing well 
# on our data because of distribution of the output column which is future_return
#  so now we are going to train classifiers on this data 
# which will use risk label as output column

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# from sklearn.metrics import (accuracy_score,precision_score,recall_score,
#     f1_score,classification_report,confusion_matrix)
# import joblib
# df=pd.read_parquet("data/model_data.parquet")
# df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
# df["future_volatility"] = (
#     df.groupby("ticker")["future_return"]
#       .transform(lambda x: x.rolling(5).std().shift(-4))
# )
# df["risk_category"] = pd.qcut(df["future_volatility"],q=4,labels=["Low","Medium","High","Very High"])
# df = df.dropna(subset=["future_volatility", "risk_category"]).reset_index(drop=True)

# x = df.drop(columns=["future_return","ticker","date","risk_category","future_volatility"])
# y = pd.Series(
#     le.fit_transform(df["risk_category"]),
#     index=df.index,
#     name="risk_category"
# )

# df = df.dropna().reset_index(drop=True)
# split = int(len(df) * 0.8)
# split = int(len(df) * 0.8)
# x_train = x.iloc[:split]
# x_test = x.iloc[split:]
# y_train = y.iloc[:split]
# y_test = y.iloc[split:]

# print("*********************** ","Train test split done"," **********************")

# models={
#     "LR": LogisticRegression(max_iter=1000,random_state=42),
#     "RF":RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1),
#     "LGBM":LGBMClassifier(n_estimators=300,random_state=42),
#     "Catboost":CatBoostClassifier(loss_function="MultiClass",random_seed=42,verbose=0),
#     "XGB":XGBClassifier(random_state=42,n_estimators=300, objective="multi:softprob",
#     eval_metric="mlogloss")
# }
# best_score = float("-inf")
# best_model = None
# best_model_name = None
# for model_name,model in models.items():
#     print(f"Training  of {model_name} started .....................")
#     model.fit(x_train,y_train)
#     y_pred=model.predict(x_test)
#     score=accuracy_score(y_test,y_pred)
#     print("Precision: ",precision_score(y_test,y_pred,average="weighted"))
#     print("Recall: ",recall_score(y_test,y_pred,average="weighted"))
#     print("Confusion matrix:",confusion_matrix(y_test,y_pred))
#     print(classification_report(y_test,y_pred,target_names=le.classes_))
#     print("model: ",model_name)
#     print("accuracy: ",score)
#     print("*****************************")
#     if score>best_score:
#         best_score=score
#         best_model=model
#         best_model_name=model_name

# print("="*50)
# print(f"Best Model : {best_model_name}")
# print(f"Accuracy   : {best_score:.4f}")



# importance = pd.DataFrame({
#     "Feature": x.columns,
#     "Importance": best_model.feature_importances_
# })

# importance = importance.sort_values("Importance",ascending=False)
# print(importance.head(20))

# joblib.dump(best_model,"models/best_classifier_model.pkl")
# joblib.dump(le, "models/label_encoder.pkl")
# print("Both label encoder and Best ML model saved successfully............")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib

le = LabelEncoder()

df = pd.read_parquet("data/model_data.parquet")
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

df["future_volatility"] = (
    df.groupby("ticker")["future_return"]
      .transform(lambda x: x.rolling(5).std().shift(-4))
)
df["risk_category"] = pd.qcut(
    df["future_volatility"], q=4, labels=["Low", "Medium", "High", "Very High"]
)
df = df.dropna().reset_index(drop=True)

x = df.drop(columns=["future_return", "ticker", "date", "risk_category", "future_volatility"])
y = pd.Series(le.fit_transform(df["risk_category"]), index=df.index, name="risk_category")

split = int(len(df) * 0.8)
x_train, x_test = x.iloc[:split], x.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Train: {x_train.shape}, Test: {x_test.shape}")
print(y_train.value_counts(normalize=True).sort_index())

tscv = TimeSeriesSplit(n_splits=5)

search_spaces = {
    "LR": {
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "params": {
            "C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            "penalty": ["l2"],
            "class_weight": [None, "balanced"],
        },
        "n_iter": 10,
        "needs_scaling": True,
    },
    "RF": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [None, 4, 6, 8, 12, 16],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
        "n_iter": 25,
        "needs_scaling": False,
    },
    "LGBM": {
        "model": LGBMClassifier(random_state=42, verbose=-1),
        "params": {
            "n_estimators": [200, 300, 500, 800],
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [-1, 4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [10, 20, 30, 50],
        },
        "n_iter": 10,
        "needs_scaling": False,
    },
    "Catboost": {
        "model": CatBoostClassifier(loss_function="MultiClass", random_seed=42, verbose=0),
        "params": {
            "iterations": [300, 500, 800],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
        },
        "n_iter": 10,
        "needs_scaling": False,
    },
    "XGB": {
        "model": XGBClassifier(random_state=42, objective="multi:softprob", eval_metric="mlogloss"),
        "params": {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5, 10],
        },
        "n_iter": 15,
        "needs_scaling": False,
    },
}

scaler = StandardScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

best_score = float("-inf")
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
        scoring="accuracy",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    print(name," training........................")
    search.fit(xt, y_train)

    tuned_model = search.best_estimator_
    y_pred = tuned_model.predict(xte)

    score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"{name}: CV={search.best_score_:.4f} Holdout={score:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    print(search.best_params_)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results[name] = {"cv_score": search.best_score_, "holdout_accuracy": score, "params": search.best_params_}

    if score > best_score:
        best_score = score
        best_model = tuned_model
        best_model_name = name
        best_params = search.best_params_

for name, r in results.items():
    print(f"{name:10s} CV: {r['cv_score']:.4f}  Holdout: {r['holdout_accuracy']:.4f}")

print(best_model_name, best_params, best_score)

if hasattr(best_model, "feature_importances_"):
    importance = pd.DataFrame({"Feature": x.columns, "Importance": best_model.feature_importances_}).sort_values("Importance", ascending=False)
    print(importance.head(20))

joblib.dump(best_model, "models/best_classifier_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Everything done successfuly !...........")










