import streamlit as st
import pandas as pd
import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import LabelEncoder

# ==========================
# CONFIG
# ==========================

st.set_page_config(
    page_title="Obesity Classification",
    page_icon="🧠",
    layout="wide"
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)



# ==========================
# DARK MODE STYLE
# ==========================

st.markdown(
"""
<style>

.stApp{
    background:#0e1117;
}


html,body,[class*="css"]{
    color:white;
}


h1,h2,h3{
    color:#60a5fa !important;
}


p{
    color:#e5e7eb !important;
}


/* sidebar */

[data-testid="stSidebar"]{
    background:#111827;
}


[data-testid="stSidebar"] *{
    color:white !important;
}



/* card */

.card{

background:#1f2937;

padding:25px;

border-radius:20px;

box-shadow:
0 8px 20px rgba(0,0,0,.5);

border:1px solid #374151;

}



.big{

font-size:40px;

font-weight:bold;

color:#60a5fa;

}



/* model */

.model-card{

background:#1f2937;

padding:30px;

border-radius:20px;

border:1px solid #374151;

}



.model-title{

font-size:25px;

font-weight:bold;

color:#93c5fd;

}



.model-text{

color:#e5e7eb;

font-size:17px;

}



/* input */

input{

background:#1f2937!important;

color:white!important;

}



div[data-baseweb="select"]>div{

background:#1f2937;

color:white;

}



/* button */

.stButton button{

background:#2563eb;

color:white;

border-radius:12px;

height:45px;

font-weight:bold;

width:100%;

}



</style>
""",
unsafe_allow_html=True
)




# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():


    rf = pickle.load(
        open(
            BASE_DIR+"/model/random_forest.pkl",
            "rb"
        )
    )


    dt = pickle.load(
        open(
            BASE_DIR+"/model/decision_tree.pkl",
            "rb"
        )
    )


    scaler = pickle.load(
        open(
            BASE_DIR+"/model/scaler.pkl",
            "rb"
        )
    )


    encoder = pickle.load(
        open(
            BASE_DIR+"/model/encoder.pkl",
            "rb"
        )
    )


    features = pickle.load(
        open(
            BASE_DIR+"/model/features.pkl",
            "rb"
        )
    )


    return rf,dt,scaler,encoder,features



rf,dt,scaler,encoder,features = load_model()



df = pd.read_csv(

    BASE_DIR+
    "/dataset/ObesityDataSet_raw_and_data_sinthetic.csv"

)




# ==========================
# SIDEBAR
# ==========================

st.sidebar.title(
    "🧠 Obesity"
)


menu = st.sidebar.radio(
    "Menu",
    [
        "🏠 Dashboard",
        "📊 Dataset",
        "📈 EDA",
        "🤖 Model",
        "🔮 Prediksi"
    ]
)



# ==========================
# DASHBOARD
# ==========================


if menu=="🏠 Dashboard":


    st.title(
        "🧠 Obesity Classification"
    )


    st.write(
        "Klasifikasi tingkat obesitas menggunakan Random Forest dan Decision Tree"
    )



    a,b,c = st.columns(3)


    a.markdown(
    f"""
    <div class="card">

    <div>Total Data</div>

    <br>

    <span class="big">
    {df.shape[0]}
    </span>

    </div>
    """,
    unsafe_allow_html=True
    )


    b.markdown(
    f"""
    <div class="card">

    <div>Jumlah Fitur</div>

    <br>

    <span class="big">
    {df.shape[1]}
    </span>

    </div>
    """,
    unsafe_allow_html=True
    )


    c.markdown(
    """
    <div class="card">

    <div>Model</div>

    <br>

    <span class="big">
    2
    </span>

    </div>
    """,
    unsafe_allow_html=True
    )




    st.subheader(
        "🤖 Algoritma"
    )


    col1,col2 = st.columns(2)


    col1.markdown(
    """
    <div class="model-card">

    <div class="model-title">
    🌲 Random Forest
    </div>

    <br>

    <div class="model-text">

    Ensemble learning yang menggabungkan
    banyak decision tree untuk meningkatkan
    performa klasifikasi.

    </div>

    </div>

    """,
    unsafe_allow_html=True
    )



    col2.markdown(
    """
    <div class="model-card">

    <div class="model-title">
    🌳 Decision Tree
    </div>

    <br>

    <div class="model-text">

    Model berbentuk pohon keputusan
    berdasarkan aturan data.

    </div>

    </div>

    """,
    unsafe_allow_html=True
    )





# ==========================
# DATASET
# ==========================


elif menu=="📊 Dataset":


    st.title(
        "📊 Dataset"
    )


    st.dataframe(
        df,
        use_container_width=True
    )





# ==========================
# EDA LENGKAP
# ==========================


elif menu=="📈 EDA":


    st.title(
        "📈 Exploratory Data Analysis"
    )


    st.write(
        "Analisis pola dan karakteristik dataset"
    )



    # 1

    st.subheader(
        "📊 Distribusi Tingkat Obesitas"
    )


    fig,ax = plt.subplots(
        figsize=(10,5)
    )


    sns.countplot(
        data=df,
        x="NObeyesdad",
        ax=ax
    )


    plt.xticks(rotation=35)


    st.pyplot(fig)




    # 2

    st.subheader(
        "👥 Gender terhadap Obesitas"
    )


    fig,ax = plt.subplots(
        figsize=(10,5)
    )


    sns.countplot(
        data=df,
        x="NObeyesdad",
        hue="Gender",
        ax=ax
    )


    plt.xticks(rotation=35)


    st.pyplot(fig)




    # 3

    st.subheader(
        "🎂 Distribusi Umur"
    )


    fig,ax = plt.subplots(
        figsize=(10,5)
    )


    sns.histplot(
        data=df,
        x="Age",
        kde=True,
        ax=ax
    )


    st.pyplot(fig)





    # 4

    st.subheader(
        "⚖️ Tinggi dan Berat Badan"
    )


    fig,ax = plt.subplots(
        figsize=(10,5)
    )


    sns.scatterplot(
        data=df,
        x="Height",
        y="Weight",
        hue="NObeyesdad",
        ax=ax
    )


    st.pyplot(fig)




    # 5

    st.subheader(
        "🏃 Aktivitas Fisik"
    )


    fig,ax = plt.subplots(
        figsize=(10,5)
    )


    sns.boxplot(
        data=df,
        x="NObeyesdad",
        y="FAF",
        ax=ax
    )


    plt.xticks(rotation=35)


    st.pyplot(fig)




    # 6

    st.subheader(
        "🔥 Korelasi Fitur"
    )


    numeric=df.select_dtypes(
        include=["int64","float64"]
    )


    fig,ax = plt.subplots(
        figsize=(12,8)
    )


    sns.heatmap(
        numeric.corr(),
        annot=True,
        ax=ax
    )


    st.pyplot(fig)


# ==========================
# MODEL
# ==========================


elif menu=="🤖 Model":


    st.title(
        "🤖 Evaluasi Model Machine Learning"
    )


    data=df.copy()



    # encoding

    categorical=[

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

        le=LabelEncoder()

        data[col]=le.fit_transform(
            data[col]
        )



    target=LabelEncoder()


    data["NObeyesdad"]=target.fit_transform(
        data["NObeyesdad"]
    )



    X=data.drop(
        "NObeyesdad",
        axis=1
    )


    y=data["NObeyesdad"]



    X=scaler.transform(
        X
    )



    X_train,X_test,y_train,y_test=train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    # ======================
    # PREDIKSI MODEL
    # ======================


    rf_pred = rf.predict(
        X_test
    )


    dt_pred = dt.predict(
        X_test
    )



    rf_acc = accuracy_score(
        y_test,
        rf_pred
    )


    dt_acc = accuracy_score(
        y_test,
        dt_pred
    )




    # ======================
    # ACCURACY CARD
    # ======================


    col1,col2 = st.columns(2)



    col1.markdown(
    f"""
    <div class="card">


    <h3>🌲 Random Forest</h3>


    <br>


    <span class="big">

    {rf_acc*100:.2f}%

    </span>


    <br>

    Accuracy


    </div>

    """,
    unsafe_allow_html=True
    )




    col2.markdown(
    f"""
    <div class="card">


    <h3>🌳 Decision Tree</h3>


    <br>


    <span class="big">

    {dt_acc*100:.2f}%

    </span>


    <br>

    Accuracy


    </div>

    """,
    unsafe_allow_html=True
    )




    st.divider()



    # ======================
    # COMPARISON GRAPH
    # ======================


    st.subheader(
        "📊 Perbandingan Akurasi"
    )



    comparison=pd.DataFrame(

    {

        "Model":
        [
            "Random Forest",
            "Decision Tree"
        ],


        "Accuracy":
        [
            rf_acc,
            dt_acc
        ]

    })


    st.bar_chart(
        comparison.set_index("Model")
    )





    # ======================
    # REPORT RANDOM FOREST
    # ======================


    st.subheader(
        "📄 Classification Report Random Forest"
    )


    rf_report=pd.DataFrame(

        classification_report(

            y_test,

            rf_pred,

            output_dict=True

        )

    ).transpose()



    st.dataframe(

        rf_report,

        use_container_width=True

    )





    # ======================
    # REPORT DECISION TREE
    # ======================


    st.subheader(
        "📄 Classification Report Decision Tree"
    )


    dt_report=pd.DataFrame(

        classification_report(

            y_test,

            dt_pred,

            output_dict=True

        )

    ).transpose()



    st.dataframe(

        dt_report,

        use_container_width=True

    )





    # ======================
    # CONFUSION MATRIX RF
    # ======================


    st.subheader(
        "🔥 Confusion Matrix Random Forest"
    )


    rf_cm=confusion_matrix(

        y_test,

        rf_pred

    )



    fig,ax=plt.subplots(
        figsize=(7,5)
    )


    sns.heatmap(

        rf_cm,

        annot=True,

        fmt="d",

        ax=ax

    )


    st.pyplot(fig)





    # ======================
    # CONFUSION MATRIX DT
    # ======================


    st.subheader(
        "🔥 Confusion Matrix Decision Tree"
    )


    dt_cm=confusion_matrix(

        y_test,

        dt_pred

    )



    fig,ax=plt.subplots(
        figsize=(7,5)
    )


    sns.heatmap(

        dt_cm,

        annot=True,

        fmt="d",

        ax=ax

    )


    st.pyplot(fig)



# ==========================
# PREDIKSI
# ==========================


else:


    st.title(
        "🔮 Prediksi Tingkat Obesitas"
    )



    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )


    age = st.number_input(
        "Age",
        10,80,20
    )


    height = st.number_input(
        "Height",
        100,220,170
    )


    weight = st.number_input(
        "Weight",
        20,200,60
    )


    family = st.selectbox(
        "Family History",
        ["yes","no"]
    )


    favc = st.selectbox(
        "FAVC",
        ["yes","no"]
    )


    fcvc = st.slider(
        "Vegetable Consumption",
        1,3
    )


    ncp = st.slider(
        "Meals",
        1,4
    )


    caec = st.selectbox(
        "Snack",
        ["no","Sometimes","Frequently","Always"]
    )


    smoke = st.selectbox(
        "Smoke",
        ["yes","no"]
    )


    ch2o = st.slider(
        "Water",
        1,3
    )


    scc = st.selectbox(
        "Calories Monitor",
        ["yes","no"]
    )


    faf = st.slider(
        "Physical Activity",
        0,3
    )


    tue = st.slider(
        "Technology Usage",
        0,2
    )


    calc = st.selectbox(
        "Alcohol",
        ["no","Sometimes","Frequently"]
    )


    mtrans = st.selectbox(
        "Transportation",
        [
        "Public_Transportation",
        "Walking",
        "Automobile",
        "Motorbike",
        "Bike"
        ]
    )




    if st.button(
        "🚀 Prediksi"
    ):


        data=[[
        1 if gender=="Male" else 0,
        age,
        height,
        weight,
        1 if family=="yes" else 0,
        1 if favc=="yes" else 0,
        fcvc,
        ncp,
        1,
        1 if smoke=="yes" else 0,
        ch2o,
        1 if scc=="yes" else 0,
        faf,
        tue,
        1,
        0
        ]]


        data=scaler.transform(
            data
        )


        hasil = rf.predict(
            data
        )


        output = encoder.inverse_transform(
            hasil
        )


        st.success(
            "Hasil Prediksi : "+output[0]
        )