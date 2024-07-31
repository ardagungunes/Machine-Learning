import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from catboost import CatBoostClassifier
import sys
import time


start = time.time()
print(sys.argv[2])

with open(sys.argv[2], "rb") as file:
    loaded_data = pickle.load(file)



print(len(loaded_data["X"]), len(loaded_data["Y"]))
print(loaded_data)


X_train, X_test, y_train, y_test = train_test_split(loaded_data['X'], loaded_data['Y'], test_size=0.2, random_state=42)


if sys.argv[1] == "xgb":
    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    print("XGBoost Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, xgb_predictions)}")
    print(f"Precision: {precision_score(y_test, xgb_predictions)}")
    print(f"Recall: {recall_score(y_test, xgb_predictions)}")
    print(f"F1 Score: {f1_score(y_test, xgb_predictions)}")



    embedding = sys.argv[2].split("-")[1]

    if embedding == "xlm":
        with open("./model/xgb-roberta.pkl", "wb") as filee:
            pickle.dump(xgb_model, filee)

    elif embedding == "sbert.pkl":
        with open("./model/xgb-sbert.pkl", "wb") as filee:
            pickle.dump(xgb_model, filee)

elif sys.argv[1] == "cat":
    # CatBoost
    catboost_model = CatBoostClassifier()
    catboost_model.fit(X_train, y_train)
    catboost_predictions = catboost_model.predict(X_test)
    print("\nCatBoost Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, catboost_predictions)}")
    print(f"Precision: {precision_score(y_test, catboost_predictions)}")
    print(f"Recall: {recall_score(y_test, catboost_predictions)}")
    print(f"F1 Score: {f1_score(y_test, catboost_predictions)}")

    embedding = sys.argv[2].split("-")[1]

    if embedding == "xlm":
        with open("./model/cat-roberta.pkl", "wb") as filee:
            pickle.dump(catboost_model, filee)

    elif embedding == "sbert.pkl":
        with open("./model/cat-sbert.pkl", "wb") as filee:
            pickle.dump(catboost_model, filee)


end = time.time()

print(end - start)