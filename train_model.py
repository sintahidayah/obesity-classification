import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "ObesityDataSet_raw_and_data_sinthetic.csv"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



# LOAD DATA

df = pd.read_csv(DATA_PATH)



# =====================
# ENCODING
# =====================

label_encoders = {}


categorical = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]


for col in categorical:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col]
    )

    label_encoders[col] = le



target_encoder = LabelEncoder()


df["NObeyesdad"] = target_encoder.fit_transform(
    df["NObeyesdad"]
)



# FEATURE

X = df.drop(
    "NObeyesdad",
    axis=1
)


y = df["NObeyesdad"]



features = X.columns.tolist()



# SCALING

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)



# SPLIT

X_train,X_test,y_train,y_test = train_test_split(

    X_scaled,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# MODEL


random_forest = RandomForestClassifier(

    n_estimators=150,

    random_state=42

)



decision_tree = DecisionTreeClassifier(

    max_depth=10,

    random_state=42

)



random_forest.fit(
    X_train,
    y_train
)


decision_tree.fit(
    X_train,
    y_train
)



# ACCURACY


rf_acc = accuracy_score(
    y_test,
    random_forest.predict(X_test)
)


dt_acc = accuracy_score(
    y_test,
    decision_tree.predict(X_test)
)



print(
    "Random Forest Accuracy :",
    rf_acc
)


print(
    "Decision Tree Accuracy :",
    dt_acc
)



# SAVE


pickle.dump(
    random_forest,
    open(
        MODEL_DIR+"/random_forest.pkl",
        "wb"
    )
)



pickle.dump(
    decision_tree,
    open(
        MODEL_DIR+"/decision_tree.pkl",
        "wb"
    )
)



pickle.dump(
    scaler,
    open(
        MODEL_DIR+"/scaler.pkl",
        "wb"
    )
)



pickle.dump(
    target_encoder,
    open(
        MODEL_DIR+"/encoder.pkl",
        "wb"
    )
)



pickle.dump(
    features,
    open(
        MODEL_DIR+"/features.pkl",
        "wb"
    )
)


print(
    "MODEL BERHASIL DISIMPAN"
)