import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd

# Yüklenen model
model = load_model('titanic_survival_model')

# Streamlit app
st.title("Titanic Survival Prediction")

# Kullanıcı girişi için form oluşturma
with st.form(key='titanic_form'):
    sex = st.selectbox("Sex", ['male', 'female'])
    fare = st.number_input("Fare", min_value=0.0, step=0.1)
    age = st.number_input("Age", min_value=0, step=1)
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sibsp = st.number_input("SibSp", min_value=0, step=1)
    parch = st.number_input("Parch", min_value=0, step=1)
    embarked = st.selectbox("Embarked", ['C', 'Q', 'S'])
    submit_button = st.form_submit_button(label='Predict')

# Tahmin işlemi
if submit_button:
    # Girilen verileri DataFrame'e dönüştürme
    data = pd.DataFrame({
        'Sex': [sex],
        'Fare': [fare],
        'Age': [age],
        'Pclass': [pclass],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Embarked': [embarked]
    })
    
    # Tahmin yapma
    prediction = predict_model(model, data=data)
    
    # Tahmin sonucunu gösterme
    if prediction['Label'][0] == 1:
        st.success("The passenger is predicted to have survived the Titanic disaster.")
    else:
        st.error("The passenger is predicted to have not survived the Titanic disaster.")