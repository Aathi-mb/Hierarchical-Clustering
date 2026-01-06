import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Hierarchical Clustering App", layout="centered")

st.title("🔗 Hierarchical Clustering App")
st.write("Upload a CSV file and perform Hierarchical Clustering")

# ===============================
# Upload Dataset
# ===============================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ===============================
    # Select Numerical Columns
    # ===============================
    X = df.select_dtypes(include=[np.number])

    if X.shape[1] < 2:
        st.error("At least 2 numerical columns are required")
    else:
        X = X.fillna(X.mean())

        # ===============================
        # Scaling
        # ===============================
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ===============================
        # Dendrogram
        # ===============================
        st.subheader("Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 5))
        linked = linkage(X_scaled, method="ward")
        dendrogram(linked, ax=ax)
        st.pyplot(fig)

        # ===============================
        # Number of Clusters
        # ===============================
        k = st.slider("Select Number of Clusters", 2, 10, 3)

        # ===============================
        # Fit Model
        # ===============================
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        clusters = hc.fit_predict(X_scaled)

        df["Cluster"] = clusters

        st.subheader("Clustered Data")
        st.dataframe(df.head())

        # ===============================
        # Cluster Plot
        # ===============================
        st.subheader("Cluster Visualization")
        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=clusters
        )
        ax2.set_xlabel(X.columns[0])
        ax2.set_ylabel(X.columns[1])
        st.pyplot(fig2)
