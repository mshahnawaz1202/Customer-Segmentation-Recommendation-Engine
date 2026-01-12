# Customer Segmentation & Recommendation Engine
"""
Streamlit application that loads the online-retail dataset, performs data cleaning, 
builds an RFM model, clusters customers with K-Means, 
creates detailed personas, and provides an interactive recommendation engine.

Features:
- Premium Glassmorphism UI (CSS)
- Native Streamlit Interactive Charts
- Real-time Clustering & Persona Generation
"""

import pathlib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Configuration & Helper Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analytics Edge | Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Global CSS
custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top left, #1a1a2e, #16213e);
        color: #e2e2e2;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(26, 26, 46, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #1a1a2e;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.5);
        color: #1a1a2e;
    }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    data_path = pathlib.Path(__file__).parent / "data.csv"
    if not data_path.exists():
        st.error(f"⚠️ Data file not found at `{data_path}`. Please ensure `data.csv` is in the root directory.")
        st.stop()
    df = pd.read_csv(data_path, encoding="ISO-8859-1")
    return df

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    return df

@st.cache_data(show_spinner=False)
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
# ML Logic (Scaling & Clustering)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def scale_rfm(rfm: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(rfm)

def fit_kmeans(data: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    model.fit(data)
    return model

@st.cache_data(show_spinner=False)
def compute_cluster_products(df: pd.DataFrame, rfm_clusters: pd.Series) -> pd.DataFrame:
    df_c = df.copy()
    df_c["Cluster"] = df_c["CustomerID"].map(rfm_clusters)
    top_products = (
        df_c.groupby(["Cluster", "Description"])["Quantity"]
        .sum()
        .reset_index()
        .sort_values(["Cluster", "Quantity"], ascending=[True, False])
    )
    return top_products

# ---------------------------------------------------------------------------
# Main App Structure
# ---------------------------------------------------------------------------
def main():
    # Load Data Initial
    raw_df = load_raw_data()
    df = clean_data(raw_df)
    rfm = build_rfm(df)

    # Initial Clustering for first-run
    if "Cluster" not in rfm.columns:
        rfm_scaled = scale_rfm(rfm)
        km_model = fit_kmeans(rfm_scaled, k=4)
        rfm["Cluster"] = km_model.labels_
        st.session_state.rfm = rfm

    # --- Sidebar Navigation ---
    st.sidebar.markdown("<h2 style='text-align: center; color: #00f2fe;'>🛍️ Analytics Pro</h2>", unsafe_allow_html=True)
    st.sidebar.write("---")
    
    if "curr_page" not in st.session_state:
        st.session_state.curr_page = "🏠 Dash"

    def set_page(name):
        st.session_state.curr_page = name

    # Button Icons
    st.sidebar.button("🏠 Business Dashboard", on_click=set_page, args=("🏠 Dash",))
    st.sidebar.button("📊 Smart Segmentation", on_click=set_page, args=("📊 Seg",))
    st.sidebar.button("👥 Customer Personas", on_click=set_page, args=("👥 Pers",))
    st.sidebar.button("💡 Recommendations", on_click=set_page, args=("💡 Recs",))
    
    page = st.session_state.curr_page

    # --- Dashboard Page ---
    if page == "🏠 Dash":
        st.title("🚀 Business Overview")
        st.markdown("### Real‑Time Customer Intelligence")
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", f"{len(rfm):,}")
        m2.metric("Total Revenue", f"${rfm['Monetary'].sum():,.0f}")
        m3.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f}")
        m4.metric("Avg Monetary", f"${rfm['Monetary'].mean():,.0f}")
        
        # Interactive Charts (Native Streamlit)
        st.subheader("Interactive Summary Charts")
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("#### Frequency vs Monetary (Cluster Heat)")
            # Using st.scatter_chart for native interactivity
            st.scatter_chart(rfm, x='Frequency', y='Monetary', color='Cluster', size='AvgOrderValue')
            
        with colB:
            st.markdown("#### Top Recency Distribution")
            st.bar_chart(rfm['Recency'].value_counts().sort_index().head(50))

        st.markdown("#### Transaction Logs (Latest)")
        st.dataframe(df.head(100), width='stretch')

    # --- Segmentation Page ---
    elif page == "📊 Seg":
        st.title("🔎 Segmentation Analysis")
        
        features = rfm.drop(columns=["Cluster"]) if "Cluster" in rfm.columns else rfm
        rfm_scaled = scale_rfm(features)
        
        # K-Selection
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Model Settings")
            k_val = st.slider("Select k (Number of Clusters)", 2, 8, 4)
            km_model = fit_kmeans(rfm_scaled, k_val)
            rfm["Cluster"] = km_model.labels_
            st.session_state.rfm = rfm
            
            score = silhouette_score(rfm_scaled, rfm["Cluster"])
            st.metric("Silhouette Score", f"{score:.3f}")
            st.progress(max(0.0, score))
        
        with col2:
            st.subheader("Inertia Elbow (Matplotlib)")
            # Elbow plot with Matplotlib
            inertia = []
            ks = range(2, 9)
            for k in ks:
                km = KMeans(n_clusters=k, random_state=42, n_init="auto")
                km.fit(rfm_scaled)
                inertia.append(km.inertia_)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(ks, inertia, marker='o', color='#00f2fe')
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.set_xlabel('k', color='white')
            ax.set_ylabel('Inertia', color='white')
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Cluster Distribution Map")
        st.scatter_chart(rfm, x='Recency', y='Monetary', color='Cluster')

    # --- Personas Page ---
    elif page == "👥 Pers":
        st.title("👥 Persona Profiles")
        rfm_data = st.session_state.rfm
        persona = rfm_data.groupby("Cluster").mean().reset_index()
        
        st.subheader("Mean Metrics by Segment")
        st.dataframe(persona.style.format(precision=2).background_gradient(cmap="Blues"), width='stretch')
        
        st.write("### Segment Archetypes")
        cols = st.columns(len(persona))
        for i, row in persona.iterrows():
            cl = int(row["Cluster"])
            with cols[cl if cl < len(cols) else 0]:
                # Dynamic Titles
                if row["Monetary"] > rfm_data["Monetary"].quantile(0.8):
                    label, icon = "💎 Champions", "🔥"
                elif row["Frequency"] > rfm_data["Frequency"].quantile(0.8):
                    label, icon = "🔄 Loyalists", "✅"
                elif row["Recency"] > rfm_data["Recency"].quantile(0.8):
                    label, icon = "🧊 Hibernating", "⚠️"
                else:
                    label, icon = "🌱 Newbies", "✨"
                
                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="color:#00f2fe">{icon} {label}</h3>
                    <p><b>Cluster ID:</b> {cl}</p>
                    <p><b>Avg Spend:</b> ${row['Monetary']:,.0f}</p>
                    <p><b>Avg Freq:</b> {row['Frequency']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)

    # --- Recommendations Page ---
    elif page == "💡 Recs":
        st.title("💡 Smart Recommendations")
        rfm_data = st.session_state.rfm
        top_products = compute_cluster_products(df, rfm_data["Cluster"])
        
        c_col, r_col = st.columns([1, 2])
        with c_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            sample_ids = rfm_data.index.to_series().sample(50).tolist()
            cust_id = st.selectbox("Select Customer ID", options=sorted(sample_ids))
            n = st.slider("Suggestions", 3, 10, 5)
            
            c_cluster = int(rfm_data.loc[cust_id, 'Cluster'])
            st.success(f"Customer {cust_id} is in **Cluster {c_cluster}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with r_col:
            recs = top_products[top_products["Cluster"] == c_cluster].head(n)
            st.subheader(f"Top {n} Personal Suggestions")
            for _, r in recs.iterrows():
                with st.expander(f"📦 {r['Description']}"):
                    st.write(f"High relevance in this segment. Total units moved: **{int(r['Quantity'])}**")
                    st.progress(0.8)

if __name__ == "__main__":
    main()
