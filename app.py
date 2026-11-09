
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Credit Card Fraud Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROFESSIONAL DESIGN
# ==================================================

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
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("models/fraud_detection_model.pkl")


try:
    artifacts = load_model()

    model = artifacts["model"]
    threshold = float(artifacts["threshold"])
    features = list(artifacts["feature_names"])

except Exception as e:
    st.error(f"Unable to load trained model: {e}")
    st.info(
        "Please check whether "
        "models/fraud_detection_model.pkl exists."
    )
    st.stop()


# ==================================================
# OPTIONAL DATASET
# ==================================================

@st.cache_data
def load_local_data():
    return pd.read_csv("data/creditcard.csv")


df = None

if os.path.exists("data/creditcard.csv"):
    try:
        df = load_local_data()
    except Exception:
        df = None


# ==================================================
# CONSTANT DATASET STATISTICS
# ==================================================

TOTAL_TRANSACTIONS = 284807
NORMAL_TRANSACTIONS = 284315
FRAUD_TRANSACTIONS = 492

FRAUD_RATE = (
    FRAUD_TRANSACTIONS / TOTAL_TRANSACTIONS
) * 100


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("💳 Fraud Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Model Analytics",
        "Fraud Prediction",
        "Dataset"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Credit Card Fraud Detection\n\n"
    "Machine Learning Project\n\n"
    "Model: Random Forest"
)

st.sidebar.markdown(
    "Developed using Python and Streamlit"
)


# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.title("Credit Card Fraud Detection")
    st.markdown(
        "### Machine Learning Analytics Dashboard"
    )

    st.write(
        "Monitor transaction patterns and explore "
        "fraud detection model performance."
    )

    st.markdown("---")

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{TOTAL_TRANSACTIONS:,}"
    )

    col2.metric(
        "Normal Transactions",
        f"{NORMAL_TRANSACTIONS:,}"
    )

    col3.metric(
        "Fraud Transactions",
        f"{FRAUD_TRANSACTIONS:,}"
    )

    col4.metric(
        "Fraud Percentage",
        f"{FRAUD_RATE:.2f}%"
    )

    st.markdown("---")

    st.subheader("Transaction Distribution")

    col1, col2 = st.columns(2)

    with col1:

        transaction_data = pd.DataFrame({
            "Transaction Type": [
                "Normal",
                "Fraud"
            ],
            "Count": [
                NORMAL_TRANSACTIONS,
                FRAUD_TRANSACTIONS
            ]
        })

        fig = px.pie(
            transaction_data,
            names="Transaction Type",
            values="Count",
            hole=0.55,
            title="Normal vs Fraud Transactions"
        )

        fig.update_layout(
            height=380,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        model_metrics = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],
            "Score": [
                99.96,
                93.18,
                83.67,
                88.17
            ]
        })

        fig = px.bar(
            model_metrics,
            x="Metric",
            y="Score",
            text="Score",
            title="Random Forest Model Performance"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=380,
            yaxis_range=[0, 110],
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Transaction Amount Analysis")

    if df is not None and "Amount" in df.columns:

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                df,
                x="Amount",
                nbins=50,
                title="Transaction Amount Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.box(
                df,
                x="Class",
                y="Amount",
                color="Class",
                title="Transaction Amount by Class"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Upload the original creditcard.csv file "
            "in the Dataset page to view transaction "
            "amount analysis."
        )

    st.caption(
        "Dataset summary figures describe the original "
        "284,807-transaction dataset. Model scores are "
        "the previously recorded test-set results."
    )


# ==================================================
# MODEL ANALYTICS PAGE
# ==================================================

elif page == "Model Analytics":

    st.title("Model Analytics")
    st.write(
        "Detailed evaluation of the trained "
        "Random Forest classification model."
    )

    st.markdown("---")

    # MODEL PERFORMANCE

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "99.96%")
    col2.metric("Precision", "93.18%")
    col3.metric("Recall", "83.67%")
    col4.metric("F1 Score", "88.17%")

    st.markdown("---")

    # MODEL COMPARISON

    st.subheader("Model Performance Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy": [
            99.91,
            99.96
        ],
        "Precision": [
            6.20,
            93.18
        ],
        "Recall": [
            91.84,
            83.67
        ],
        "F1 Score": [
            11.62,
            88.17
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        comparison,
        x="Model",
        y=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        barmode="group",
        title="Classification Model Comparison"
    )

    fig.update_layout(
        yaxis_range=[0, 110],
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # CONFUSION MATRIX

    st.subheader("Confusion Matrix")

    confusion_matrix = [
        [56858, 6],
        [16, 82]
    ]

    fig = px.imshow(
        confusion_matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        x=["Predicted Normal", "Predicted Fraud"],
        y=["Actual Normal", "Actual Fraud"],
        title="Random Forest Confusion Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # THRESHOLD ANALYSIS

    st.subheader("Prediction Threshold Analysis")

    threshold_data = pd.DataFrame({
        "Threshold": [
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9
        ],
        "F1 Score": [
            78, 85, 88.17, 87, 85,
            82, 78, 72, 60
        ]
    })

    fig = px.line(
        threshold_data,
        x="Threshold",
        y="F1 Score",
        markers=True,
        title="F1 Score Across Prediction Thresholds"
    )

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text="Selected Threshold"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        f"The trained model uses a prediction threshold "
        f"of {threshold:.2f}."
    )

    st.caption(
        "Evaluation figures above are recorded test results. "
        "The comparison and threshold charts are illustrative "
        "unless recalculated from your experiment outputs."
    )


# ==================================================
# FRAUD PREDICTION PAGE
# ==================================================

elif page == "Fraud Prediction":

    st.title("Fraud Prediction")
    st.write(
        "Upload transaction data to predict "
        "whether transactions are normal or fraudulent."
    )

    st.markdown("---")

    st.subheader("Upload Transaction CSV")

    st.write(
        "Your CSV must contain all the model input "
        "features. The optional Class column is ignored."
    )

    # CREATE CSV TEMPLATE

    template = pd.DataFrame(
        [[0] * len(features)],
        columns=features
    )

    st.download_button(
        label="Download CSV Template",
        data=template.to_csv(index=False),
        file_name="fraud_prediction_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Choose transaction CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            input_df = pd.read_csv(uploaded_file)

            # REMOVE OPTIONAL TARGET COLUMN

            if "Class" in input_df.columns:
                input_df = input_df.drop(
                    columns=["Class"]
                )

            # CHECK REQUIRED FEATURES

            missing_features = [
                col for col in features
                if col not in input_df.columns
            ]

            if missing_features:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_features)
                )

            else:

                prediction_df = input_df[features].copy()

                prediction_df = prediction_df.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if prediction_df.isnull().any().any():

                    st.error(
                        "Your CSV contains missing or "
                        "non-numeric feature values. "
                        "Please correct the file."
                    )

                else:

                    st.subheader("Uploaded Transaction Data")

                    st.dataframe(
                        prediction_df.head(10),
                        use_container_width=True
                    )

                    if st.button(
                        "Submit & Predict Fraud",
                        use_container_width=True
                    ):

                        probabilities = model.predict_proba(
                            prediction_df
                        )[:, 1]

                        predictions = (
                            probabilities >= threshold
                        ).astype(int)

                        results = prediction_df.copy()

                        results["Fraud Probability"] = (
                            probabilities * 100
                        ).round(2)

                        results["Prediction"] = [
                            "Fraud" if value == 1
                            else "Normal"
                            for value in predictions
                        ]

                        st.markdown("---")

                        st.subheader("Prediction Results")

                        total = len(results)
                        fraud_count = int(
                            (predictions == 1).sum()
                        )
                        normal_count = total - fraud_count

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Total Predictions",
                            total
                        )

                        col2.metric(
                            "Fraud Detected",
                            fraud_count
                        )

                        col3.metric(
                            "Normal Transactions",
                            normal_count
                        )

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        # RESULT CHART

                        result_counts = pd.DataFrame({
                            "Transaction Type": [
                                "Normal",
                                "Fraud"
                            ],
                            "Count": [
                                normal_count,
                                fraud_count
                            ]
                        })

                        fig = px.pie(
                            result_counts,
                            names="Transaction Type",
                            values="Count",
                            hole=0.5,
                            title="Prediction Distribution"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # DOWNLOAD RESULTS

                        st.download_button(
                            label="Download Prediction Results",
                            data=results.to_csv(index=False),
                            file_name="fraud_prediction_results.csv",
                            mime="text/csv"
                        )

        except Exception as e:

            st.error(
                f"Unable to process the uploaded file: {e}"
            )


# ==================================================
# DATASET PAGE
# ==================================================

elif page == "Dataset":

    st.title("Dataset Explorer")

    st.write(
        "Explore the credit card transaction dataset."
    )

    st.markdown("---")

    if df is not None:

        st.success(
            "Original dataset loaded successfully."
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader("Search Dataset Columns")

        selected_column = st.selectbox(
            "Select a column",
            df.columns
        )

        st.write(
            "Selected column:",
            selected_column
        )

        st.dataframe(
            df[[selected_column]].head(20),
            use_container_width=True
        )

        st.download_button(
            "Download Dataset",
            data=df.to_csv(index=False),
            file_name="creditcard_dataset.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "The original dataset is not available "
            "in this environment."
        )

        st.write(
            "Upload creditcard.csv to explore the dataset."
        )

        dataset_file = st.file_uploader(
            "Upload Original Dataset",
            type=["csv"],
            key="dataset_upload"
        )

        if dataset_file is not None:

            try:

                uploaded_df = pd.read_csv(dataset_file)

                st.success(
                    "Dataset uploaded successfully!"
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Rows",
                    f"{uploaded_df.shape[0]:,}"
                )

                col2.metric(
                    "Columns",
                    uploaded_df.shape[1]
                )

                st.subheader("Dataset Preview")

                st.dataframe(
                    uploaded_df.head(100),
                    use_container_width=True
                )

                st.download_button(
                    "Download Uploaded Dataset",
                    data=uploaded_df.to_csv(index=False),
                    file_name="uploaded_creditcard_dataset.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    f"Unable to read dataset: {e}"
                )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748B;">
        Credit Card Fraud Detection |
        Machine Learning Project |
        Built with Python and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
