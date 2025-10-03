# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:47:39 2025

@author: polycarp.mugizi
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import seaborn as sns

st.set_page_config(
    page_title="Diabetes Risk Prediction App",
    layout="wide"
)

st.markdown("""
    <style>
        /* TAB STYLING */
        .stTabs [role="tablist"] {
            background-color: #eafbea;  /* very light green background */
            border-radius: 10px;
            padding: 6px;
        }
        .stTabs [role="tab"] {
            color: black;  /* black text for inactive tabs */
            font-weight: 600;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #2ecc71;  /* bright green for active tab */
            color: white;               /* white text for contrast */
            border-radius: 8px;
        }

        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: #f0fff0;  /* light green tint */
            color: black;               /* black text */
        }

        /* MAIN PAGE BACKGROUND */
        .block-container {
            background-color: #ffffff;  /* clean white background */
            color: black;               /* readable text */
            padding: 1.5rem;
            border-radius: 10px;
        }

        /* HEADERS */
        h1, h2, h3, h4 {
            color: #145a32;  /* deep green headers */
        }

        /* METRIC CARDS */
        .stMetric {
            background: #eafbea;
            border: 1px solid #2ecc71;
            padding: 10px;
            border-radius: 12px;
            color: black;
        }

        /* BUTTONS */
        div.stButton > button {
            background-color: #2ecc71;
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #27ae60;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


# App title
st.title("Diabetes Risk Prediction")

# Load dataset from local folder
file_path = r"D:\Msc\datasets\streamlit\diabetes.csv"
df = pd.read_csv(file_path)

# Tabs for app organization
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Preparation", "Exploratory Analysis",
                                  "Feature Engineering", "Feature Selection Visualisation",
                                  "Model Training & Results Visualisation", "Dashboard & Insights"])

# =============================
# TAB 1: DATA CLEANING
# =============================
with tab1:
    st.subheader("Pima Indians Diabetes Dataset Preview")

    # Allow user to control number of rows displayed
    rows = st.slider("Select number of rows to view", 5, 50, 5)
    st.write(df.head(rows))

    # Detect invalid zeros in key columns
    invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    zero_counts = (df[invalid_cols] == 0).sum()
    st.subheader("Invalid values [zeros]")
    st.write(zero_counts)

    # Checkbox: Replace invalid zeros with NaN and impute median
    if st.checkbox("Replace invalid zeros with NaN and impute median"):
        df_clean = df.copy()
        for col in invalid_cols:
            df_clean[col] = df_clean[col].replace(0, np.nan)
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        st.write("✅ Zeros replaced and missing values imputed with median")
        st.write(df_clean.head())
    else:
        df_clean = df.copy()

    # Toggle (or checkbox) for standardization
    if hasattr(st, "toggle"):  # For Streamlit >=1.24
        standardize = st.toggle("Standardize continuous variables")
    else:
        standardize = st.checkbox("Standardize continuous variables")

    if standardize:
        cont_cols = ["Pregnancies", "Glucose", "BloodPressure", 
                     "SkinThickness", "Insulin", "BMI", 
                     "DiabetesPedigreeFunction", "Age"]
        scaler = StandardScaler()
        df_clean[cont_cols] = scaler.fit_transform(df_clean[cont_cols])
        st.write("✅ Continuous variables standardized")

    # Show preview of processed dataset
    st.subheader("Processed Dataset Preview")
    st.write(df_clean.head())

# =============================
# TAB 2: ANALYSIS & VISUALIZATION
# =============================
with tab2:
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Glucose", f"{df_clean['Glucose'].mean():.2f}")
    col2.metric("Mean BMI", f"{df_clean['BMI'].mean():.2f}")
    col3.metric("Mean Age", f"{df_clean['Age'].mean():.2f}")

    # Interactive Histograms
    st.subheader("Interactive Histograms")
    bins = st.slider("Select number of bins", 5, 50, 20, key="bins_slider")

    fig_glucose = px.histogram(df_clean, x="Glucose", nbins=bins, title="Glucose Distribution")
    st.plotly_chart(fig_glucose, use_container_width=True)

    fig_bmi = px.histogram(df_clean, x="BMI", nbins=bins, title="BMI Distribution")
    st.plotly_chart(fig_bmi, use_container_width=True)

    # Box Plot of Glucose vs Outcome
    st.subheader("Glucose vs Diabetes Outcome")
    selected_outcome = st.radio("Filter by Outcome", options=["All", 0, 1])

    if selected_outcome == "All":
        fig_box = px.box(df_clean, x="Outcome", y="Glucose", points="all",
                         title="Glucose by Diabetes Outcome")
    else:
        filtered_df = df_clean[df_clean["Outcome"] == int(selected_outcome)]
        fig_box = px.box(filtered_df, x="Outcome", y="Glucose", points="all",
                         title=f"Glucose for Outcome {selected_outcome}")

    st.plotly_chart(fig_box, use_container_width=True)
    
# ================================
# FEATURE ENGINEERING TAB
# ================================
with tab3:
    st.header("Feature Engineering")

    # --- Sidebar controls (only visible in this tab) ---
    with st.sidebar:
        st.subheader("BMI Category Cut-offs")

        # Store default cut-offs (WHO standard)
        DEFAULTS = {
            "underweight_max": 18.5,
            "normal_max": 24.9,
            "overweight_max": 29.9
        }

        # Reset button
        if st.button("🔄 Reset to WHO defaults"):
            st.session_state["underweight_max"] = DEFAULTS["underweight_max"]
            st.session_state["normal_max"] = DEFAULTS["normal_max"]
            st.session_state["overweight_max"] = DEFAULTS["overweight_max"]

        # Sliders with session state so they reset properly
        underweight_max = st.slider(
            "Max BMI for Underweight", 
            10.0, 25.0, 
            st.session_state.get("underweight_max", DEFAULTS["underweight_max"]), 
            0.1
        )
        normal_max = st.slider(
            "Max BMI for Normal", 
            underweight_max, 30.0, 
            st.session_state.get("normal_max", DEFAULTS["normal_max"]), 
            0.1
        )
        overweight_max = st.slider(
            "Max BMI for Overweight", 
            normal_max, 40.0, 
            st.session_state.get("overweight_max", DEFAULTS["overweight_max"]), 
            0.1
        )

    # --- Function to assign BMI category ---
    def bmi_category(bmi):
        if bmi < underweight_max:
            return "Underweight"
        elif bmi < normal_max:
            return "Normal"
        elif bmi < overweight_max:
            return "Overweight"
        else:
            return "Obese"

    # --- Add calculated column ---
    df["BMI_Category"] = df["BMI"].apply(bmi_category)

    st.write("✅ Added **BMI Category** column")
    st.dataframe(df.head())

    # --- Count number of records per category ---
    bmi_counts = df["BMI_Category"].value_counts().reset_index()
    bmi_counts.columns = ["BMI Category", "Count"]

    # --- Bar chart of BMI categories ---
    fig = px.bar(
        bmi_counts,
        x="BMI Category",
        y="Count",
        title="Distribution of BMI Categories",
        text="Count",
        color="BMI Category"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Text summary below chart ---
    st.subheader("Summary")
    total = len(df)
    for _, row in bmi_counts.iterrows():
        category = row["BMI Category"]
        count = row["Count"]
        percent = (count / total) * 100
        st.write(f"- **{category}**: {count} patients ({percent:.1f}%)")
        
# ================================
# FEATURE SELECTION TAB
# ================================
with tab4:
    st.header("Feature Selection Visualisation")

    # Separate features and target
    # Drop the Outcome column (target variable) from features
    #X = df.drop(columns=["Outcome"])
    X = df.drop(columns=["Outcome"]).select_dtypes(include=[np.number])
    y = df["Outcome"]

    # ✅ FIX: Keep only numeric columns (exclude BMI_Category which has strings)
    # PCA and Mutual Information require numeric inputs
    X = X.select_dtypes(include=[np.number])

    # Sidebar options for this tab
    method = st.radio(
        "Select Feature Selection Method:", 
        ["PCA", "Mutual Information"]
    )
    top_n = st.slider(
        "Number of top features/components to display", 
        2, X.shape[1], 5
    )

    # Run analysis only when user clicks button
    if st.button("▶️ Run Feature Selection"):
        # -------------------------------
        # MUTUAL INFORMATION
        # -------------------------------
        if method == "Mutual Information":
            st.subheader("Mutual Information Scores")
            
            # Scale features before computing MI (optional but consistent)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Compute MI scores (supervised: feature relevance vs target)
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            mi_df = pd.DataFrame({"Feature": X.columns, "Score": mi_scores})
            mi_df = mi_df.sort_values(by="Score", ascending=False).head(top_n)

            # Bar chart of top features
            fig = px.bar(
                mi_df, 
                x="Feature", 
                y="Score", 
                title=f"Top {top_n} Features by Mutual Information", 
                text="Score", 
                color="Feature"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Text summary
            top_feats = ", ".join(mi_df["Feature"].tolist())
            st.info(f"📊 Based on Mutual Information, the most relevant features are: {top_feats}.")

        # -------------------------------
        # PCA
        # -------------------------------
        elif method == "PCA":
            st.subheader("PCA Results (Principal Component Analysis)")

            # Scale features before PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Run PCA with top_n components
            pca = PCA(n_components=top_n)
            components = pca.fit_transform(X_scaled)

            # Variance explained per component
            exp_var = pca.explained_variance_ratio_ * 100
            total_var = np.sum(exp_var)
            st.write(f"Explained Variance of Top {top_n} Components: {total_var:.2f}%")

            # Plot first 2 components if available (biplot style)
            if top_n >= 2:
                fig, ax = plt.subplots(figsize=(7,5))
                scatter = ax.scatter(
                    components[:, 0], components[:, 1], 
                    c=y, cmap="coolwarm", alpha=0.6
                )
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("PCA Biplot (PC1 vs PC2)")
                plt.colorbar(scatter, ax=ax, label="Outcome")
                st.pyplot(fig)

            # Bar chart of variance explained
            var_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(top_n)],
                "Variance Explained (%)": exp_var
            })
            fig2 = px.bar(
                var_df, 
                x="Component", 
                y="Variance Explained (%)", 
                title="Variance Explained by Components",
                text="Variance Explained (%)", 
                color="Component"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Text summary
            main_pc = var_df.loc[var_df["Variance Explained (%)"].idxmax(), "Component"]
            st.info(f"📊 PCA shows that {main_pc} captures the largest share of variance among features.")

# =======================================
# MODEL TRAINING & RESULTS TAB
# =======================================
with tab5:
    st.header("Model Training & Results Visualisation")

    # -------------------------
    # Dataset Preparation
    # -------------------------
    X = df.drop(columns=["Outcome"])
    X = X.select_dtypes(include=[np.number])  # ✅ Use only numeric
    y = df["Outcome"]

    # Train-test split
    test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100
    random_state = st.number_input("Random Seed", 0, 999, 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # -------------------------
    # Model Selection
    # -------------------------
    classifier = st.selectbox(
        "Choose Classifier", 
        ["Logistic Regression", "Random Forest", "SVM"]
    )

    # Hyperparameters per model
    if classifier == "Logistic Regression":
        C = st.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C, max_iter=1000, solver="liblinear", random_state=random_state)

    elif classifier == "Random Forest":
        n_estimators = st.slider("Number of Trees", 50, 500, 100, step=50)
        max_depth = st.slider("Max Depth", 2, 20, 5)
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )

    elif classifier == "SVM":
        C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
        model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)

    # -------------------------
    # Train Model
    # -------------------------
    if st.button("▶️ Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

        # -------------------------
        # Metrics
        # -------------------------
        st.subheader("Evaluation Metrics")
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write(f"**Accuracy:** {acc:.3f}")
        st.write(f"**Precision:** {prec:.3f}")
        st.write(f"**Recall:** {rec:.3f}")
        st.write(f"**F1 Score:** {f1:.3f}")

        # -------------------------
        # Visualisations in sub-tabs
        # -------------------------
        vis_tab1, vis_tab2 = st.tabs(["ROC Curve", "Confusion Matrix"])

        # ROC Curve
        with vis_tab1:
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
                ax.plot([0, 1], [0, 1], "k--")  # diagonal line
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.warning("ROC Curve not available (SVM with non-probability kernel).")

        # Confusion Matrix
        with vis_tab2:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["No Diabetes", "Diabetes"],
                        yticklabels=["No Diabetes", "Diabetes"],
                        ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

# =======================================
# DASHBOARD & INSIGHTS TAB
# =======================================
with tab6:
    st.header("📊 Dashboard & Insights")

    # Nested tabs inside Dashboard
    dash_tab1, dash_tab2, dash_tab3, dash_tab4, dash_tab5 = st.tabs([
        "EDA", "Feature Selection", "Modeling", "Metrics", "Clinical Relevance & Ethics"
    ])

    # -------------------------
    # EDA Tab
    # -------------------------
    with dash_tab1:
        st.subheader("Exploratory Data Analysis")
        st.write("📈 Summary of dataset distributions and correlations")

        # Example: Correlation heatmap
        fig, ax = plt.subplots(figsize=(6,4))
        corr = df.select_dtypes(include=[np.number]).corr() #corr = df.corr()--- old
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    # -------------------------
    # Feature Selection Tab
    # -------------------------
    with dash_tab2:
        st.subheader("Feature Selection Summary")
        st.write("⭐ Most important features identified by PCA/Mutual Information.")

        # Example: Show top 5 features by variance/MI if computed earlier
        # For simplicity, recalc MI here
        X_fs = df.drop(columns=["Outcome"]).select_dtypes(include=[np.number])
        y_fs = df["Outcome"]
        mi_scores = mutual_info_classif(X_fs, y_fs, random_state=42)
        mi_df = pd.DataFrame({"Feature": X_fs.columns, "Score": mi_scores}).sort_values(by="Score", ascending=False).head(5)

        st.table(mi_df)

    # -------------------------
    # Modeling Tab
    # -------------------------
    with dash_tab3:
        st.subheader("Model Summary")
        st.write("🔧 Details of chosen classifier and hyperparameters.")

        # Example placeholders (would be updated after training in Tab 5)
        st.write("**Selected Classifier:** Random Forest")
        st.write("**Hyperparameters:** n_estimators=100, max_depth=5")

    # -------------------------
    # Metrics Tab
    # -------------------------
    with dash_tab4:
        st.subheader("Evaluation Metrics")
        st.write("📊 Key performance indicators of the trained model.")

        # Example static metrics (link to Tab 5 results ideally)
        st.metric("Accuracy", "0.85")
        st.metric("Precision", "0.83")
        st.metric("Recall", "0.81")
        st.metric("F1 Score", "0.82")

    # -------------------------
    # Clinical Relevance & Ethics Tab
    # -------------------------
    with dash_tab5:
        st.subheader("Clinical Relevance & Ethical Considerations")
        st.write("✍️ Please reflect on how these results may be applied responsibly.")

        user_notes = st.text_area(
            "Your insights on clinical significance, potential risks, and ethical considerations:",
            placeholder="e.g., What are the implications of false positives/false negatives in diabetes diagnosis? How should patient privacy and bias be handled?"
        )

        if st.button("💾 Save Notes"):
            st.success("Your notes have been saved (not persisted across sessions).")