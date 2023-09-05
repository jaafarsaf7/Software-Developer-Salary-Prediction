import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")


    st.write("""### We need some informations to predict the salary""")
    countries = ("United States of America", 
             "Germany ",
             "United Kingdom of Great Britain and Northern Ireland",
             "Canada",
             "India",
             "France",
             "Netherlands",
             "Australia",
             "Brazil",
             "Spain",
             "Sweden",
             "Italy",
    )

    education = (
    "Bachelor’s degree",
    "less than a Bachelor",
    "Master’s degree",
    "Post grad",
    )


    country = st.selectbox("County", countries)
    education = st.selectbox("Educaion", education)

    expericence = st.slider("years of Experience", 0, 50, 3)

    start = st.button("Predict Salary")
    if start:
        X = np.array([[country, education, expericence]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        salary = regressor_loaded.predict(X)
        st.subheader(f"The estimated salary is {salary[0]:}$")



st.markdown("---")
st.markdown('<p style="font-size:16px;color:blue;">Created by Jaafar safar</p>', unsafe_allow_html=True)





