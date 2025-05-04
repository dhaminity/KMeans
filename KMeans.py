import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set Streamlit to use a custom port
os.environ['STREAMLIT_SERVER_PORT'] = '8070'

# Ensure directories exist
os.makedirs("data/input", exist_ok=True)  # Make sure the input directory exists
os.makedirs("data/test", exist_ok=True)   # Make sure the test directory exists

st.set_page_config(page_title="KMeans Clustering UI", layout="wide")

# Sidebar navigation
menu = ["Input Data", "Fine-tune Parameters", "Run Model", "View & Download Output"]
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", menu)

# Shared session state
if 'df_train' not in st.session_state:
    st.session_state.df_train = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_columns' not in st.session_state:
    st.session_state.train_columns = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Input Data
if choice == "Input Data":
    st.title("Step 1: Upload Training Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_train = df
        df.to_csv("data/input/train.csv", index=False)
        st.success("Training data uploaded successfully.")
        st.dataframe(df.head(10))

# Fine-tune Parameters
elif choice == "Fine-tune Parameters":
    st.title("Step 2: Fine-tune Parameters")
    if st.session_state.df_train is not None:
        df = st.session_state.df_train

        st.subheader("Set KMeans Parameters")
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        init_method = st.selectbox("Init Method", ["k-means++", "random"])
        max_iter = st.slider("Max Iterations", 100, 1000, 300, step=50)
        n_init = st.slider("Number of Initializations", 5, 20, 10)
        random_state = st.slider("Random State", 0, 100, 42)
        tol = st.slider("Tolerance for Convergence", 1e-4, 1e-2, 1e-4, step=1e-5)

        if st.button("Train Model"):
            features = ['Education', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'AmtWines',
                        'AmtFruits', 'AmtMeatProducts', 'AmtFishProducts', 'AmtSweetProducts',
                        'AmtGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
                        'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4',
                        'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response', 'AmtOrganic',
                        'AmtNonOrganic', 'AmtReadytoEat', 'AmtCookedFoods', 'AmtEatable', 'AmtNonEatable',
                        'AmtCosmetic', 'Tenure_Days', 'Tenure_Months', 'Tenure_Years', 'Marital_Status_Divorced',
                        'Marital_Status_Married', 'Marital_Status_Single', 'Marital_Status_Together',
                        'Marital_Status_Widow', 'Age']

            data = df[features]

            # Scaling the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Train KMeans model with added 'tol' parameter
            model = KMeans(n_clusters=n_clusters, init=init_method, max_iter=max_iter,
                           n_init=n_init, random_state=random_state, tol=tol)
            model.fit(data_scaled)

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.features = features

            st.success("KMeans model trained successfully!")

    else:
        st.warning("Please upload training data in 'Input Data' section.")

# Run Model
elif choice == "Run Model":
    st.title("Step 3: Run Model on Test Data")
    if st.session_state.model is not None:
        test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test")
        if test_file:
            df_test = pd.read_csv(test_file)

            # Apply the same scaling as in training
            scaler = st.session_state.scaler
            features = st.session_state.features
            X_test = df_test[features]
            X_test_scaled = scaler.transform(X_test)

            # Predict clusters
            clusters = st.session_state.model.predict(X_test_scaled)

            # Add predicted cluster column to test data
            label_map = {0: "High Spender", 1: "Moderate Spender", 2: "Low Spender"}
            df_test['Predicted Spender Type'] = [label_map[cluster] for cluster in clusters]

            # Display top 10 records with the predicted column
            st.session_state.predictions = df_test
            st.success("Prediction completed.")
            st.dataframe(df_test[['Predicted Spender Type'] + features].head(10))

            # Add download link for full predictions
            # Ensure the predicted column is included in the download file
            csv = df_test.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(
                f"""
                <a href="data:file/csv;base64,{b64}" download="predicted_output.csv"
                   style="color: blue; text-decoration: underline; font-size: 16px;">
                   📥 Download Full Predicted File
                </a>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("Please train the model in 'Fine-tune Parameters' section.")

# View & Download Output
elif choice == "View & Download Output":
    st.title("Step 4: Visualize & Download Results")

    if st.session_state.predictions is not None:
        df_pred = st.session_state.predictions

        # Display the predicted cluster distribution
        st.subheader("Cluster Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_pred['Predicted Spender Type'].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title("Predicted Spender Type Distribution")
        ax.set_ylabel('Count')
        ax.set_xlabel('Spender Type')
        st.pyplot(fig)

    else:
        st.warning("Please run the model in the 'Run Model' section.")
