import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Pakistan Air Quality ML", layout="wide")

st.title("Pakistan Air Quality Prediction App")

file = st.file_uploader("Upload Air Quality Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # encode categorical columns
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    # handle missing values
    df = df.fillna(df.mean())

    st.subheader("Clean Dataset")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if target:

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor()

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)

        st.success(f"Model R2 Score: {r2}")

        st.subheader("Data Visualization")

        st.bar_chart(df[target])

        st.line_chart(df[target])

        st.subheader("Prediction")

        inputs = []

        for col in X.columns:
            val = st.number_input(col)
            inputs.append(val)

        if st.button("Predict"):

            prediction = model.predict([inputs])

            st.success(f"Predicted {target}: {prediction[0]}")
    