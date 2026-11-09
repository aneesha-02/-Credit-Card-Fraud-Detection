
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Credit Card Fraud Analytics",
    page_icon="💳",
    layout="wide"
)

# ---------------- PROFESSIONAL DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
    color: #172033;
}
[data-testid="stHeader"] {
    background-color: #F4F7FB;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #172033 !important;
}
[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] {
    color: #64748B;
}
[data-testid="stMetricValue"] {
    color: #1D4ED8;
}
div[data-testid="stSidebar"] {
    background-color: #172B4D;
}
div[data-testid="stSidebar"] * {
    color: white !important;
}
.stButton > button,
.stDownloadButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}
hr {
    border-color: #E2E8F0;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA AND MODEL ----------------

@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/fraud_detection_model.pkl")


try:
    df = load_data()
    artifacts = load_model()

    model = artifacts["model"]
    threshold = artifacts["threshold"]
    features = artifacts["feature_names"]

except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()


# ---------------- SIDEBAR ----------------

st.sidebar.title("💳 Fraud Analytics")
st.sidebar.caption("Machine Learning System")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Model Analytics",
        "Fraud Prediction",
        "Dataset"
    ]
)

st.sidebar.divider()
st.sidebar.info("Model: Random Forest Classifier")
st.sidebar.caption("Prediction Threshold: 0.30")


# ---------------- CHART THEME ----------------

chart_layout = {
    "template": "plotly_white",
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "font": {"color": "#172033"},
    "margin": dict(l=30, r=30, t=60, b=30)
}


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.title("Credit Card Fraud Detection")
    st.caption("Transaction Monitoring & Analytics Dashboard")

    st.divider()

    total = len(df)
    fraud = int(df["Class"].sum())
    normal = total - fraud
    fraud_rate = fraud / total * 100

    # KPI CARDS

    st.subheader("Transaction Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Normal Transactions", f"{normal:,}")
    c3.metric("Fraud Transactions", f"{fraud:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    st.divider()

    # TRANSACTION CHARTS

    st.subheader("Transaction Analytics")

    left, right = st.columns(2)

    with left:

        class_data = pd.DataFrame({
            "Transaction Type": ["Normal", "Fraud"],
            "Count": [normal, fraud]
        })

        fig = px.pie(
            class_data,
            names="Transaction Type",
            values="Count",
            hole=0.65,
            title="Transaction Distribution",
            color="Transaction Type",
            color_discrete_map={
                "Normal": "#2563EB",
                "Fraud": "#F59E0B"
            }
        )

        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, width="stretch")

    with right:

        amount_data = df.copy()
        amount_data["Transaction Type"] = amount_data["Class"].map({
            0: "Normal",
            1: "Fraud"
        })

        fig = px.box(
            amount_data,
            x="Transaction Type",
            y="Amount",
            color="Transaction Type",
            title="Transaction Amount Analysis",
            color_discrete_map={
                "Normal": "#2563EB",
                "Fraud": "#F59E0B"
            }
        )

        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Transaction Amount Distribution")

    fig = px.histogram(
        df,
        x="Amount",
        color="Class",
        nbins=50,
        title="Normal vs Fraud Transaction Amounts",
        color_discrete_map={
            0: "#2563EB",
            1: "#F59E0B"
        },
        labels={"Class": "Transaction Type"}
    )

    fig.update_layout(**chart_layout)
    st.plotly_chart(fig, width="stretch")


# ==================================================
# MODEL ANALYTICS
# ==================================================

elif page == "Model Analytics":

    st.title("Model Analytics")
    st.caption("Performance evaluation of machine learning models")

    st.divider()

    st.subheader("Final Random Forest Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "99.96%")
    c2.metric("Precision", "93.18%")
    c3.metric("Recall", "83.67%")
    c4.metric("F1 Score", "88.17%")

    st.caption(
        "Test-set performance using Random Forest with a prediction threshold of 0.30."
    )

    st.divider()

    # MODEL COMPARISON

    st.subheader("Model Performance Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest (0.5)",
            "Random Forest (0.3)"
        ],
        "Accuracy": [97.55, 99.95, 99.96],
        "Precision": [6.10, 96.00, 93.18],
        "Recall": [91.84, 73.47, 83.67],
        "F1 Score": [11.44, 83.24, 88.17]
    })

    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch"
    )

    fig = px.bar(
        comparison.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score"
        ),
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Machine Learning Model Comparison",
        text_auto=".2f"
    )

    fig.update_layout(**chart_layout)
    fig.update_yaxes(title="Score (%)", range=[0, 105])

    st.plotly_chart(fig, width="stretch")

    st.divider()

    # CONFUSION MATRIX

    st.subheader("Confusion Matrix")

    st.write(
        "Final Random Forest model results at the selected threshold of 0.30."
    )

    cm = [
        [56858, 6],
        [16, 82]
    ]

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted Normal", "Predicted Fraud"],
        y=["Actual Normal", "Actual Fraud"],
        color_continuous_scale="Blues",
        title="Random Forest Confusion Matrix"
    )

    fig.update_layout(**chart_layout)
    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Threshold Analysis")

    threshold_data = pd.DataFrame({
        "Threshold": [0.5, 0.4, 0.3, 0.2, 0.1],
        "Precision": [96.00, 93.83, 93.18, 85.86, 75.65],
        "Recall": [73.47, 77.55, 83.67, 86.73, 88.78],
        "F1 Score": [83.24, 84.92, 88.17, 86.29, 81.69]
    })

    fig = px.line(
        threshold_data,
        x="Threshold",
        y=["Precision", "Recall", "F1 Score"],
        markers=True,
        title="Impact of Prediction Threshold"
    )

    fig.update_layout(**chart_layout)
    fig.update_yaxes(title="Score (%)")

    st.plotly_chart(fig, width="stretch")

    st.success(
        "Selected threshold: 0.30 — it achieved the highest F1 score "
        "among the tested thresholds."
    )


# ==================================================
# FRAUD PREDICTION
# ==================================================

elif page == "Fraud Prediction":

    st.title("Fraud Prediction")
    st.caption("Upload transaction data and identify possible fraudulent transactions.")

    st.divider()

    st.info(
        "Upload a CSV file containing all 30 required features: "
        "Time, V1 to V28, and Amount."
    )

    # DOWNLOAD TEMPLATE

    template = pd.DataFrame([
        {feature: 0.0 for feature in features}
    ])

    st.download_button(
        "Download CSV Template",
        data=template.to_csv(index=False),
        file_name="transaction_template.csv",
        mime="text/csv"
    )

    # SAMPLE DATA

    sample = df.sample(
        n=min(100, len(df)),
        random_state=42
    ).drop(columns=["Class"])

    st.download_button(
        "Download Sample Transactions",
        data=sample.to_csv(index=False),
        file_name="sample_transactions.csv",
        mime="text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            input_data = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Transaction Preview")

            st.dataframe(
                input_data.head(),
                width="stretch"
            )

            with st.form("prediction_form"):

                st.write("Ready to analyze the uploaded transactions?")

                submitted = st.form_submit_button(
                    "Submit & Predict Fraud",
                    width="stretch"
                )

            if submitted:

                missing = [
                    col for col in features
                    if col not in input_data.columns
                ]

                if missing:

                    st.error(f"Missing required columns: {missing}")

                else:

                    X_input = input_data[features].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    if X_input.isnull().any().any():

                        st.error(
                            "Please check the CSV. All feature values "
                            "must be numeric and contain no missing values."
                        )

                    else:

                        probabilities = model.predict_proba(
                            X_input
                        )[:, 1]

                        predictions = (
                            probabilities >= threshold
                        ).astype(int)

                        results = input_data.copy()

                        results["Fraud Probability (%)"] = (
                            probabilities * 100
                        ).round(2)

                        results["Prediction"] = [
                            "Fraud" if p == 1 else "Normal"
                            for p in predictions
                        ]

                        st.divider()

                        st.subheader("Prediction Results")

                        fraud_count = int(predictions.sum())
                        normal_count = len(predictions) - fraud_count

                        a, b = st.columns(2)

                        a.metric(
                            "Fraud Detected",
                            fraud_count
                        )

                        b.metric(
                            "Normal Transactions",
                            normal_count
                        )

                        st.dataframe(
                            results,
                            width="stretch"
                        )

                        st.download_button(
                            "Download Prediction Results",
                            data=results.to_csv(index=False),
                            file_name="fraud_prediction_results.csv",
                            mime="text/csv"
                        )

                        if fraud_count > 0:
                            st.warning(
                                f"{fraud_count} possible fraudulent "
                                "transaction(s) detected."
                            )
                        else:
                            st.success(
                                "No possible fraudulent transactions detected."
                            )

        except Exception as e:

            st.error(f"Unable to process file: {e}")


# ==================================================
# DATASET
# ==================================================

elif page == "Dataset":

    st.title("Dataset Explorer")
    st.caption("Explore the credit card transaction dataset.")

    st.divider()

    c1, c2 = st.columns(2)

    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])

    search = st.text_input("Search for a column name")

    if search:

        matching = [
            col for col in df.columns
            if search.lower() in col.lower()
        ]

        if matching:
            st.dataframe(
                df[matching].head(100),
                width="stretch"
            )
        else:
            st.warning("No matching column found.")

    else:

        st.dataframe(
            df.head(100),
            width="stretch"
        )

    st.download_button(
        "Download Dataset",
        data=df.to_csv(index=False),
        file_name="credit_card_dataset.csv",
        mime="text/csv"
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Credit Card Fraud Detection | Random Forest Machine Learning Project"
)
