import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd
#from joblib import load
# PyCaret modelini yükle
model = load_model("titanic_pycaret_model")
#model=load('titanic_pycaret_model.pkl')
# Web uygulaması arayüzünü oluştur
st.title("Titanic Survived Prediction")
st.subheader("@drmurataltun-2024")

# Kullanıcı girişlerini al
sex = st.radio("Sex", ['female', 'male'], index=1)
age = st.slider("Age", min_value=0, max_value=100, value=22)
fare = st.slider("Fare (British pounds)", min_value=0, max_value=1000, value=15.0,step=1)
Pclass = st.radio("Travel Class", ['1', '2', '3'], index=0)
Embarked = st.radio("Embarked", ['C', 'Q', 'S'], index=2)
SibSp=st.number_input(label='Kardeş/Eş sayısı',min_value=0,max_value=10, step=1)
Parch=st.number_input(label='Ebeveyn/Çocuk sayısı',min_value=0,max_value=10, step=1)


# Tahmin yap ve sonucu göster
if st.button("Predict"):
    # Giriş verilerini DataFrame'e dönüştür
    data = pd.DataFrame({
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare],
        'Pclass': [Pclass],
        'Embarked': [Embarked],
        'SibSp':[SibSp],
        'Parch':[Parch],

    })
    
    # Modele veriyi gönder ve sonucu al
    result = predict_model(model, data=data)
    
    # Sonucu göster
    if result['prediction_label'][0] == 1:
        st.write("Kurtulur")
    else:
        st.write("Kurtulamaz")
