# ================================================================
# PLANT LEAF PATTERN & SIMILARITY DISCOVERY
# Unsupervised Machine Learning Project
# ================================================================

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Plant Leaf Pattern Discovery",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #888888;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #444444;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================================================================
# TITLE
# ================================================================

st.markdown(
    '<div class="main-title">🌿 Plant Leaf Pattern & Similarity Discovery</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Unsupervised discovery of visual patterns and similarity groups '
    'among plant leaf images'
    '</div>',
    unsafe_allow_html=True
)


# ================================================================
# DEVICE
# ================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# LOAD RESNET50
# ================================================================

@st.cache_resource
def load_feature_extractor():

    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)

    # Remove final classification layer
    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    transform = weights.transforms()

    return model, transform


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def extract_features(image_list, model, transform):

    features = []
    valid_images = []
    names = []

    progress = st.progress(0)

    for i, item in enumerate(image_list):

        try:

            image = Image.open(item).convert("RGB")

            tensor = transform(image)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device)

            with torch.no_grad():

                feature = model(tensor)

            feature = feature.cpu().numpy().flatten()

            features.append(feature)
            valid_images.append(image.copy())

            if hasattr(item, "name"):
                names.append(item.name)
            else:
                names.append(str(item))

        except Exception:
            pass

        progress.progress(
            (i + 1) / len(image_list)
        )

    progress.empty()

    return (
        np.array(features),
        valid_images,
        names
    )


# ================================================================
# DATASET FROM LOCAL FOLDER
# ================================================================

def load_folder_images(folder_path):

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    image_paths = []

    if not os.path.exists(folder_path):
        return []

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            if file.lower().endswith(valid_extensions):

                image_paths.append(
                    os.path.join(root, file)
                )

    return image_paths


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title("Project Controls")

st.sidebar.markdown(
    """
    Configure the unsupervised learning pipeline.
    """
)


# ================================================================
# DATASET INPUT
# ================================================================

st.sidebar.subheader("1. Dataset")

uploaded_files = st.sidebar.file_uploader(
    "Upload plant leaf images",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],
    accept_multiple_files=True
)


# Also check local dataset folder

dataset_folder = "dataset"

local_image_paths = load_folder_images(
    dataset_folder
)


# ================================================================
# COMBINE DATA SOURCES
# ================================================================

image_sources = []

if uploaded_files:

    image_sources.extend(uploaded_files)


if local_image_paths:

    image_sources.extend(local_image_paths)


# ================================================================
# NO DATASET
# ================================================================

if len(image_sources) == 0:

    st.info(
        "Upload plant leaf images from the sidebar "
        "or place images inside the 'dataset' folder."
    )

    st.markdown(
        """
        ## How the system works

        **Plant Leaf Images**

        ↓

        **ResNet50 Feature Extraction**

        ↓

        **Standardization**

        ↓

        **PCA Dimensionality Reduction**

        ↓

        **K-Means / DBSCAN**

        ↓

        **Cluster Evaluation**

        ↓

        **Similarity Search**
        """
    )

    st.stop()


# ================================================================
# DATASET INFORMATION
# ================================================================

st.sidebar.success(
    f"{len(image_sources)} image(s) loaded"
)


# ================================================================
# LOAD MODEL
# ================================================================

with st.spinner("Loading ResNet50 feature extractor..."):

    model, transform = load_feature_extractor()


# ================================================================
# EXTRACT FEATURES
# ================================================================

with st.spinner(
    "Extracting visual features from leaf images..."
):

    features, images, image_names = extract_features(
        image_sources,
        model,
        transform
    )


# ================================================================
# VALIDATION
# ================================================================

n_samples = len(features)


if n_samples < 2:

    st.error(
        "At least 2 valid images are required."
    )

    st.stop()


if n_samples < 10:

    st.warning(
        f"Only {n_samples} images are available. "
        "For meaningful clustering and evaluation, "
        "use at least 10 images. 50+ images are recommended."
    )


# ================================================================
# STANDARDIZATION
# ================================================================

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    features
)


# ================================================================
# PCA FOR CLUSTERING
# ================================================================

pca_components = min(
    50,
    n_samples - 1,
    scaled_features.shape[1]
)

if pca_components >= 2:

    pca_model = PCA(
        n_components=pca_components,
        random_state=42
    )

    reduced_features = pca_model.fit_transform(
        scaled_features
    )

else:

    reduced_features = scaled_features


# ================================================================
# SIDEBAR K-MEANS CONTROL
# ================================================================

st.sidebar.subheader("2. K-Means")


max_k = min(
    15,
    n_samples - 1
)


if max_k < 2:

    st.error(
        "Not enough images for K-Means clustering."
    )

    st.stop()


# IMPORTANT:
# Avoid Streamlit slider error when min == max.

if max_k == 2:

    k_value = 2

    st.sidebar.info(
        "K automatically set to 2 because the dataset is small."
    )

else:

    default_k = min(
        4,
        max_k
    )

    k_value = st.sidebar.slider(
        "Number of clusters (K)",
        min_value=2,
        max_value=max_k,
        value=default_k,
        step=1
    )


# ================================================================
# DBSCAN CONTROLS
# ================================================================

st.sidebar.subheader("3. DBSCAN")

dbscan_eps = st.sidebar.slider(
    "eps",
    min_value=0.1,
    max_value=5.0,
    value=0.8,
    step=0.1
)


dbscan_max_samples = min(
    10,
    n_samples
)


dbscan_min_samples = st.sidebar.slider(
    "min_samples",
    min_value=2,
    max_value=dbscan_max_samples,
    value=min(3, dbscan_max_samples),
    step=1
)


# ================================================================
# RUN K-MEANS
# ================================================================

kmeans = KMeans(
    n_clusters=k_value,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(
    reduced_features
)


# ================================================================
# RUN DBSCAN
# ================================================================

dbscan = DBSCAN(
    eps=dbscan_eps,
    min_samples=dbscan_min_samples
)

dbscan_labels = dbscan.fit_predict(
    reduced_features
)


# ================================================================
# PCA 2D FOR VISUALIZATION
# ================================================================

if n_samples >= 3:

    pca_2d = PCA(
        n_components=2,
        random_state=42
    )

    visualization_features = pca_2d.fit_transform(
        scaled_features
    )

else:

    visualization_features = scaled_features[:, :2]


# ================================================================
# EVALUATION FUNCTION
# ================================================================

def calculate_metrics(
    data,
    labels
):

    unique_labels = set(labels)

    # Remove DBSCAN noise for metric calculation
    valid_mask = labels != -1

    clean_data = data[valid_mask]
    clean_labels = labels[valid_mask]

    n_clusters = len(
        set(clean_labels)
    )

    if n_clusters < 2:

        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "clusters": n_clusters,
            "noise": int(np.sum(labels == -1))
        }

    if len(clean_labels) <= n_clusters:

        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "clusters": n_clusters,
            "noise": int(np.sum(labels == -1))
        }

    try:

        silhouette = silhouette_score(
            clean_data,
            clean_labels
        )

    except Exception:

        silhouette = np.nan


    try:

        db_score = davies_bouldin_score(
            clean_data,
            clean_labels
        )

    except Exception:

        db_score = np.nan


    try:

        ch_score = calinski_harabasz_score(
            clean_data,
            clean_labels
        )

    except Exception:

        ch_score = np.nan


    return {
        "silhouette": silhouette,
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
        "clusters": n_clusters,
        "noise": int(np.sum(labels == -1))
    }


# ================================================================
# CALCULATE METRICS
# ================================================================

kmeans_metrics = calculate_metrics(
    reduced_features,
    kmeans_labels
)


dbscan_metrics = calculate_metrics(
    reduced_features,
    dbscan_labels
)


# ================================================================
# TOP METRICS
# ================================================================

st.markdown(
    '<div class="section-title">Dataset Overview</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Images",
        n_samples
    )


with col2:

    st.metric(
        "CNN Features",
        features.shape[1]
    )


with col3:

    st.metric(
        "K-Means Clusters",
        k_value
    )


with col4:

    if np.isnan(kmeans_metrics["silhouette"]):

        score_text = "N/A"

    else:

        score_text = (
            f"{kmeans_metrics['silhouette']:.3f}"
        )

    st.metric(
        "K-Means Silhouette",
        score_text
    )


# ================================================================
# TABS
# ================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "K-Means",
        "DBSCAN",
        "Evaluation",
        "Similar Leaf Search"
    ]
)


# ================================================================
# TAB 1 — OVERVIEW
# ================================================================

with tab1:

    st.header(
        "Visual Pattern Discovery"
    )

    st.write(
        "The following visualization shows the plant leaf "
        "feature space after dimensionality reduction."
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    scatter = ax.scatter(
        visualization_features[:, 0],
        visualization_features[:, 1],
        c=kmeans_labels,
        cmap="tab10",
        s=80,
        alpha=0.85
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        "Plant Leaf Patterns Discovered by K-Means"
    )

    ax.grid(
        alpha=0.2
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    st.subheader(
        "Pipeline"
    )

    st.markdown(
        """
        **Leaf Images**

        ↓

        **ResNet50 Feature Extraction**

        ↓

        **Standardization**

        ↓

        **PCA**

        ↓

        **K-Means / DBSCAN**

        ↓

        **Cluster Evaluation**

        ↓

        **Similar Leaf Retrieval**
        """
    )


# ================================================================
# TAB 2 — K-MEANS
# ================================================================

with tab2:

    st.header(
        "K-Means Clustering"
    )

    st.write(
        f"K-Means discovered {k_value} visual groups."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        value = kmeans_metrics["silhouette"]

        st.metric(
            "Silhouette Score",
            "N/A" if np.isnan(value)
            else f"{value:.4f}"
        )


    with col2:

        value = kmeans_metrics["davies_bouldin"]

        st.metric(
            "Davies-Bouldin Index",
            "N/A" if np.isnan(value)
            else f"{value:.4f}"
        )


    with col3:

        value = kmeans_metrics["calinski_harabasz"]

        st.metric(
            "Calinski-Harabasz Score",
            "N/A" if np.isnan(value)
            else f"{value:.2f}"
        )


    st.subheader(
        "K-Means Visualization"
    )


    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    ax.scatter(
        visualization_features[:, 0],
        visualization_features[:, 1],
        c=kmeans_labels,
        cmap="tab10",
        s=80,
        alpha=0.85
    )


    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        f"K-Means Clustering (K={k_value})"
    )

    ax.grid(
        alpha=0.2
    )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ------------------------------------------------------------
    # CLUSTER TABLE
    # ------------------------------------------------------------

    cluster_df = pd.DataFrame(
        {
            "Image": image_names,
            "Cluster": kmeans_labels
        }
    )


    st.subheader(
        "Cluster Assignment"
    )

    st.dataframe(
        cluster_df,
        use_container_width=True
    )


# ================================================================
# TAB 3 — DBSCAN
# ================================================================

with tab3:

    st.header(
        "DBSCAN Clustering"
    )

    st.write(
        "DBSCAN identifies density-based groups and can mark "
        "isolated samples as noise."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        value = dbscan_metrics["silhouette"]

        st.metric(
            "Silhouette Score",
            "N/A" if np.isnan(value)
            else f"{value:.4f}"
        )


    with col2:

        value = dbscan_metrics["davies_bouldin"]

        st.metric(
            "Davies-Bouldin Index",
            "N/A" if np.isnan(value)
            else f"{value:.4f}"
        )


    with col3:

        st.metric(
            "Noise Points",
            dbscan_metrics["noise"]
        )


    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    ax.scatter(
        visualization_features[:, 0],
        visualization_features[:, 1],
        c=dbscan_labels,
        cmap="tab10",
        s=80,
        alpha=0.85
    )


    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        "DBSCAN Clustering"
    )

    ax.grid(
        alpha=0.2
    )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ================================================================
# TAB 4 — EVALUATION
# ================================================================

with tab4:

    st.header(
        "Clustering Evaluation"
    )


    evaluation_data = {

        "Method": [
            "K-Means",
            "DBSCAN"
        ],

        "Clusters": [
            kmeans_metrics["clusters"],
            dbscan_metrics["clusters"]
        ],

        "Noise": [
            kmeans_metrics["noise"],
            dbscan_metrics["noise"]
        ],

        "Silhouette": [
            kmeans_metrics["silhouette"],
            dbscan_metrics["silhouette"]
        ],

        "Davies-Bouldin": [
            kmeans_metrics["davies_bouldin"],
            dbscan_metrics["davies_bouldin"]
        ],

        "Calinski-Harabasz": [
            kmeans_metrics["calinski_harabasz"],
            dbscan_metrics["calinski_harabasz"]
        ]
    }


    evaluation_df = pd.DataFrame(
        evaluation_data
    )


    st.dataframe(
        evaluation_df,
        use_container_width=True
    )


    st.subheader(
        "Metric Interpretation"
    )


    st.markdown(
        """
        **Silhouette Score**

        Higher values generally indicate better-separated clusters.

        **Davies-Bouldin Index**

        Lower values generally indicate better clustering.

        **Calinski-Harabasz Score**

        Higher values generally indicate better-defined clusters.

        **DBSCAN Noise**

        Samples labelled `-1` are considered noise/outliers.
        """
    )


    # ------------------------------------------------------------
    # BEST METHOD
    # ------------------------------------------------------------

    k_score = kmeans_metrics["silhouette"]
    d_score = dbscan_metrics["silhouette"]


    if not np.isnan(k_score) and not np.isnan(d_score):

        if k_score >= d_score:

            best_method = "K-Means"

        else:

            best_method = "DBSCAN"

        st.success(
            f"Best clustering method based on Silhouette Score: "
            f"**{best_method}**"
        )

    elif not np.isnan(k_score):

        st.success(
            "K-Means produced the valid Silhouette Score."
        )

    elif not np.isnan(d_score):

        st.success(
            "DBSCAN produced the valid Silhouette Score."
        )

    else:

        st.warning(
            "A valid Silhouette Score could not be calculated. "
            "Try increasing the number of images or adjusting DBSCAN parameters."
        )


# ================================================================
# TAB 5 — SIMILARITY SEARCH
# ================================================================

with tab5:

    st.header(
        "🔎 Similar Leaf Search"
    )

    st.write(
        "Upload a new leaf image and retrieve visually similar "
        "images from the dataset."
    )


    query_file = st.file_uploader(
        "Upload a query leaf image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ],
        key="query_leaf"
    )


    if query_file is not None:

        try:

            query_image = Image.open(
                query_file
            ).convert("RGB")


            st.subheader(
                "Query Leaf"
            )


            st.image(
                query_image,
                width=250
            )


            # ----------------------------------------------------
            # QUERY FEATURE
            # ----------------------------------------------------

            query_tensor = transform(
                query_image
            )

            query_tensor = query_tensor.unsqueeze(
                0
            )

            query_tensor = query_tensor.to(
                device
            )


            with torch.no_grad():

                query_feature = model(
                    query_tensor
                )


            query_feature = (
                query_feature
                .cpu()
                .numpy()
                .flatten()
                .reshape(1, -1)
            )


            # ----------------------------------------------------
            # COSINE SIMILARITY
            # ----------------------------------------------------

            similarities = cosine_similarity(
                query_feature,
                features
            )[0]


            # Don't recommend the exact same image if it exists
            ranked_indices = np.argsort(
                similarities
            )[::-1]


            top_n = min(
                5,
                len(ranked_indices)
            )


            top_indices = ranked_indices[
                :top_n
            ]


            st.subheader(
                "Most Similar Leaves"
            )


            cols = st.columns(
                top_n
            )


            for position, index in enumerate(
                top_indices
            ):

                with cols[position]:

                    st.image(
                        images[index],
                        use_container_width=True
                    )

                    similarity_percent = (
                        similarities[index] * 100
                    )

                    st.markdown(
                        f"**{similarity_percent:.2f}% similar**"
                    )

                    st.caption(
                        image_names[index]
                    )


            # ----------------------------------------------------
            # QUERY CLUSTER
            # ----------------------------------------------------

            query_scaled = scaler.transform(
                query_feature
            )


            query_reduced = pca_model.transform(
                query_scaled
            )


            query_cluster = kmeans.predict(
                query_reduced
            )[0]


            st.info(
                f"The query image is closest to "
                f"**K-Means Cluster {query_cluster}**."
            )


        except Exception as e:

            st.error(
                f"Could not process the query image: {e}"
            )


# ================================================================
# DOWNLOAD RESULTS
# ================================================================

st.sidebar.subheader(
    "4. Export"
)


results_df = pd.DataFrame(
    {
        "Image": image_names,
        "KMeans_Cluster": kmeans_labels,
        "DBSCAN_Cluster": dbscan_labels
    }
)


csv_data = results_df.to_csv(
    index=False
).encode("utf-8")


st.sidebar.download_button(
    label="Download Cluster Results",
    data=csv_data,
    file_name="plant_leaf_clustering_results.csv",
    mime="text/csv"
)


# ================================================================
# FOOTER
# ================================================================

st.markdown(
    "---"
)

st.markdown(
    """
    <div style="text-align:center; color:#888888;">
    Plant Leaf Pattern & Similarity Discovery |
    Unsupervised Machine Learning |
    ResNet50 + K-Means + DBSCAN + PCA
    </div>
    """,
    unsafe_allow_html=True
)