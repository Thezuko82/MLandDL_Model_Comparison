import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Title and description
st.title("ML and DL Model Comparison for Regression")
st.markdown("Upload your dataset with 4 independent variables and 1 dependent variable.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    if df.shape[1] != 5:
        st.error("Dataset must have exactly 4 independent variables and 1 dependent variable.")
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Data splitting and scaling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model Selection
        st.sidebar.title("Select Models to Compare")
        models_selected = st.sidebar.multiselect("Choose ML/DL models", [
            "Linear Regression", "Decision Tree", "Random Forest", "SVM", "Deep Learning"
        ], default=["Linear Regression", "Decision Tree"])

        r2_scores = {}
        mse_scores = {}

        # Linear Regression
        if "Linear Regression" in models_selected:
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred_lr = lr.predict(X_test_scaled)
            r2_scores['Linear Regression'] = r2_score(y_test, y_pred_lr)
            mse_scores['Linear Regression'] = mean_squared_error(y_test, y_pred_lr)

        # Decision Tree
        if "Decision Tree" in models_selected:
            dt = DecisionTreeRegressor()
            dt.fit(X_train, y_train)
            y_pred_dt = dt.predict(X_test)
            r2_scores['Decision Tree'] = r2_score(y_test, y_pred_dt)
            mse_scores['Decision Tree'] = mean_squared_error(y_test, y_pred_dt)

        # Random Forest
        if "Random Forest" in models_selected:
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            r2_scores['Random Forest'] = r2_score(y_test, y_pred_rf)
            mse_scores['Random Forest'] = mean_squared_error(y_test, y_pred_rf)

        # SVM
        if "SVM" in models_selected:
            svr = SVR()
            svr.fit(X_train_scaled, y_train)
            y_pred_svr = svr.predict(X_test_scaled)
            r2_scores['SVM'] = r2_score(y_test, y_pred_svr)
            mse_scores['SVM'] = mean_squared_error(y_test, y_pred_svr)

        # Deep Learning
        if "Deep Learning" in models_selected:
            model = Sequential()
            model.add(Dense(64, input_dim=4, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_scaled, y_train, epochs=100, verbose=0)
            y_pred_dl = model.predict(X_test_scaled).flatten()
            r2_scores['Deep Learning'] = r2_score(y_test, y_pred_dl)
            mse_scores['Deep Learning'] = mean_squared_error(y_test, y_pred_dl)

        # Show performance
        if r2_scores and mse_scores:
            r2_df = pd.DataFrame(list(r2_scores.items()), columns=['Model', 'R2 Score'])
            mse_df = pd.DataFrame(list(mse_scores.items()), columns=['Model', 'MSE'])

            st.write("### Model Performance")
            st.write("#### R2 Scores")
            st.dataframe(r2_df.set_index("Model"))
            st.write("#### Mean Squared Errors")
            st.dataframe(mse_df.set_index("Model"))

            # R2 Score Plot
            st.write("### R2 Score Comparison")
            fig_r2, ax_r2 = plt.subplots()
            sns.barplot(x='Model', y='R2 Score', data=r2_df, ax=ax_r2)
            plt.xticks(rotation=45)
            plt.title("R2 Score Comparison")
            st.pyplot(fig_r2)

            # MSE Plot
            st.write("### Mean Squared Error (MSE) Comparison")
            fig_mse, ax_mse = plt.subplots()
            sns.barplot(x='Model', y='MSE', data=mse_df, ax=ax_mse)
            plt.xticks(rotation=45)
            plt.title("MSE Comparison")
            st.pyplot(fig_mse)
else:
    st.info("Please upload a CSV file with 4 independent and 1 dependent variable.")
