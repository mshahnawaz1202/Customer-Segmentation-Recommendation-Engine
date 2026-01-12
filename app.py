# Customer Segmentation & Recommendation Engine
"""
Streamlit application that loads the online‑retail dataset, performs data cleaning, builds an RFM model, clusters customers with K‑Means, creates personas, and provides product recommendations.

Features
--------
- **Data Overview** – preview raw data and summary statistics.
- **Customer Segmentation** – elbow plot to choose K, cluster distribution visualisation.
- **Customer Personas** – mean RFM values per cluster with explanatory markdown.
- **Recommendation Engine** – pick a CustomerID and see top‑N product suggestions.

The app uses **pandas**, **scikit‑learn**, **matplotlib**, and **streamlit**.  Place the CSV file (e.g. `online_retail.csv`) inside a `data/` folder next to this script.
"""

import pathlib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration & Helper Styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Customer Segmentation & Recommendation", layout="wide")

# Custom CSS for a premium look (dark mode, glass‑morphism style)
custom_css = """
    <style>
    .main {background: linear-gradient(135deg, #1e1e2f, #2a2a3d); color: #f0f0f0;}
    .stButton>button {background: rgba(255,255,255,0.1); color: #fff; border-radius: 8px; border: 1px solid #555;}
    .stSelectbox>div>div>div {color: #fff;}
    .stDataFrame {background: rgba(255,255,255,0.05); border-radius: 8px;}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV file. The file should be placed as `data.csv` in the project root.
    Returns a DataFrame with the original columns.
    """
    data_path = pathlib.Path(__file__).parent / "data.csv"
    if not data_path.exists():
        st.error(f"Data file not found at `{data_path}`. Please add `data.csv`.")
        st.stop()
    df = pd.read_csv(data_path, encoding="ISO-8859-1")
    return df

# ---------------------------------------------------------------------------
# Step 1 – Data Cleaning
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning steps described in the specification.
    1. Remove rows with missing CustomerID
    2. Drop cancelled invoices (InvoiceNo starts with 'C')
    3. Keep only positive quantities
    4. Convert InvoiceDate to datetime
    """
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])  # safety net
    return df

# ---------------------------------------------------------------------------
# Step 2 – Feature Engineering (RFM)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Create the RFM table and the AvgOrderValue column.
    Returns a DataFrame indexed by CustomerID with columns Recency, Frequency, Monetary, AvgOrderValue.
    """
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    return rfm

# ---------------------------------------------------------------------------
# Step 3 – Scaling & Clustering
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def scale_rfm(rfm: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(rfm)

def fit_kmeans(data: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    model.fit(data)
    return model

# ---------------------------------------------------------------------------
# Step 4 – Cluster‑Product Matrix & Recommendation Logic
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_cluster_products(df: pd.DataFrame, rfm_clusters: pd.Series) -> pd.DataFrame:
    """Return a DataFrame with columns: Cluster, Description, Quantity (total quantity purchased).
    """
    df = df.copy()
    df["Cluster"] = rfm_clusters
    top_products = (
        df.groupby(["Cluster", "Description"])["Quantity"]
        .sum()
        .reset_index()
        .sort_values(["Cluster", "Quantity"], ascending=[True, False])
    )
    return top_products

def recommend_products(customer_id: int, rfm: pd.DataFrame, top_products: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if customer_id not in rfm.index:
        return pd.DataFrame()
    cluster = rfm.loc[customer_id, "Cluster"]
    recs = top_products[top_products["Cluster"] == cluster].head(n)
    return recs[["Description", "Quantity"]]

# ---------------------------------------------------------------------------
# UI Helper Functions
# ---------------------------------------------------------------------------
def plot_elbow(rfm_scaled: np.ndarray, max_k: int = 10):
    inertia = []
    ks = range(2, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(rfm_scaled)
        inertia.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(ks, inertia, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for optimal k")
    st.pyplot(fig)

from sklearn.metrics import silhouette_score

def plot_rfm_distributions(rfm: pd.DataFrame):
    """Plot boxplots of RFM values by Cluster."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["Recency", "Frequency", "Monetary"]
    for i, metric in enumerate(metrics):
        axes[i].boxplot([rfm[rfm["Cluster"] == c][metric] for c in sorted(rfm["Cluster"].unique())])
        axes[i].set_title(f"{metric} by Cluster")
        axes[i].set_xticklabels(sorted(rfm["Cluster"].unique()))
    st.pyplot(fig)

def plot_cluster_distribution(rfm: pd.DataFrame):
    counts = rfm["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values, color="#4e79a7")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of customers")
    ax.set_title("Cluster distribution")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    st.title("🛍️ Customer Segmentation & Recommendation Engine")
    # Load and prepare data (cached)
    raw_df = load_raw_data()
    df = clean_data(raw_df)
    rfm = build_rfm(df)

    # Initial clustering so other pages work immediately
    if "Cluster" not in rfm.columns:
        # Scale only features (not the ID or existing clusters)
        rfm_scaled = scale_rfm(rfm)
        km_model = fit_kmeans(rfm_scaled, k=4) # Default to 4 clusters
        rfm["Cluster"] = km_model.labels_

    # Sidebar navigation – button based
    if "page" not in st.session_state:
        st.session_state.page = "Data Overview"
    # Create buttons
    if st.sidebar.button("Data Overview"):
        st.session_state.page = "Data Overview"
    if st.sidebar.button("Customer Segmentation"):
        st.session_state.page = "Customer Segmentation"
    if st.sidebar.button("Customer Personas"):
        st.session_state.page = "Customer Personas"
    if st.sidebar.button("Recommendation Engine"):
        st.session_state.page = "Recommendation Engine"
    page = st.session_state.page

    if page == "Data Overview":
        st.header("📊 Data Overview")
        st.subheader("Raw data preview (first 5 rows)")
        st.dataframe(raw_df.head())
        st.subheader("Cleaned data preview (first 5 rows)")
        st.dataframe(df.head())
        st.subheader("Summary statistics")
        st.write(df.describe())

    elif page == "Customer Segmentation":
        st.header("🔎 Customer Segmentation")
        st.subheader("Elbow Plot – choose optimal K")
        
        # EXCLUDE 'Cluster' from scaling if it exists
        features = rfm.drop(columns=["Cluster"]) if "Cluster" in rfm.columns else rfm
        rfm_scaled = scale_rfm(features)
        
        plot_elbow(rfm_scaled, max_k=10)
        # Let user pick K (default 4 as in the spec)
        k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=4)
        km_model = fit_kmeans(rfm_scaled, k)
        rfm["Cluster"] = km_model.labels_
        
        # Silhouette Score calculation
        score = silhouette_score(rfm_scaled, rfm["Cluster"])
        st.metric("Silhouette Score", f"{score:.3f}")
        st.info("💡 Silhouette score validates cluster separation quality (higher is better).")

        st.success(f"K‑Means fitted with K={k}. Assigned clusters added to RFM table.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cluster counts")
            plot_cluster_distribution(rfm)
        with col2:
            st.subheader("RFM Metrics by Cluster")
            plot_rfm_distributions(rfm)

        st.subheader("Cluster centroids (scaled space)")
        cent_df = pd.DataFrame(km_model.cluster_centers_, columns=rfm.columns.drop("Cluster"))
        st.dataframe(cent_df)

    elif page == "Customer Personas":
        st.header("👥 Customer Personas")
        if "Cluster" not in rfm.columns:
            st.warning("Run the *Customer Segmentation* page first to create clusters.")
        else:
            persona = rfm.groupby("Cluster").mean().reset_index()
            st.subheader("Average RFM values per persona")
            st.dataframe(persona)
            # Simple textual explanations – you can expand these as needed
            st.subheader("Interpretation")
            for _, row in persona.iterrows():
                cl = int(row["Cluster"])
                rec = f"**Cluster {cl}** – "
                if row["Monetary"] > rfm["Monetary"].median():
                    rec += "High‑value customers. "
                else:
                    rec += "Low‑value customers. "
                if row["Frequency"] > rfm["Frequency"].median():
                    rec += "Frequent purchasers. "
                else:
                    rec += "Infrequent purchasers. "
                if row["Recency"] < rfm["Recency"].median():
                    rec += "Recent activity."
                else:
                    rec += "Long time since last purchase."
                st.markdown(rec)

    elif page == "Recommendation Engine":
        st.header("💡 Recommendation Engine")
        if "Cluster" not in rfm.columns:
            st.warning("Run the *Customer Segmentation* page first to compute clusters.")
        else:
            # Pre‑compute cluster‑product matrix (cached)
            top_products = compute_cluster_products(df, rfm["Cluster"])
            # Customer selector – show a few sample IDs for convenience
            sample_ids = rfm.index.to_series().sample(min(20, len(rfm))).sort_values().tolist()
            cust_id = st.selectbox("Select CustomerID", options=sample_ids)
            n = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
            recs = recommend_products(cust_id, rfm, top_products, n=n)
            if recs.empty:
                st.info("No recommendations found for this customer.")
            else:
                st.subheader(f"Top {n} product recommendations for Customer {cust_id}")
                st.dataframe(recs)

    else:
        st.info("Select a page from the sidebar.")

if __name__ == "__main__":
    main()
