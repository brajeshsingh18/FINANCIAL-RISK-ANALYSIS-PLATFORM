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

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
from sklearn.metrics import (accuracy_score,precision_score,recall_score,
    f1_score,classification_report,confusion_matrix)
import joblib
df=pd.read_parquet("data/model_data.parquet")
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
df["future_volatility"] = (
    df.groupby("ticker")["future_return"]
      .transform(lambda x: x.rolling(5).std().shift(-4))
)
df["risk_category"] = pd.qcut(df["future_volatility"],q=4,labels=["Low","Medium","High","Very High"])
df = df.dropna(subset=["future_volatility", "risk_category"]).reset_index(drop=True)

x = df.drop(columns=["future_return","ticker","date","risk_category","future_volatility"])
y = pd.Series(
    le.fit_transform(df["risk_category"]),
    index=df.index,
    name="risk_category"
)

df = df.dropna().reset_index(drop=True)
split = int(len(df) * 0.8)
split = int(len(df) * 0.8)
x_train = x.iloc[:split]
x_test = x.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("*********************** ","Train test split done"," **********************")

models={
    "LR": LogisticRegression(max_iter=1000,random_state=42),
    "RF":RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1),
    "LGBM":LGBMClassifier(n_estimators=300,random_state=42),
    "Catboost":CatBoostClassifier(loss_function="MultiClass",random_seed=42,verbose=0),
    "XGB":XGBClassifier(random_state=42,n_estimators=300, objective="multi:softprob",
    eval_metric="mlogloss")
}
best_score = float("-inf")
best_model = None
best_model_name = None
for model_name,model in models.items():
    print(f"Training  of {model_name} started .....................")
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    score=accuracy_score(y_test,y_pred)
    print("Precision: ",precision_score(y_test,y_pred,average="weighted"))
    print("Recall: ",recall_score(y_test,y_pred,average="weighted"))
    print("Confusion matrix:",confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred,target_names=le.classes_))
    print("model: ",model_name)
    print("accuracy: ",score)
    print("*****************************")
    if score>best_score:
        best_score=score
        best_model=model
        best_model_name=model_name

print("="*50)
print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_score:.4f}")

importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": best_model.feature_importances_
})

importance = importance.sort_values("Importance",ascending=False)
print(importance.head(20))

joblib.dump(best_model,"models/best_classifier_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
print("Both label encoder and Best ML model saved successfully............")











