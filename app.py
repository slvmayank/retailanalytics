from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Online Retail Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Online Retail Analytics Dashboard")
st.caption(
    "Sales performance · RFM customer segments · 30-day repeat-purchase analysis"
)


# ------------------------------------------------------------
# Data preparation and model training
# ------------------------------------------------------------
@st.cache_data(show_spinner="Loading transactions and preparing dashboard...")
def prepare_dashboard(file_bytes: bytes):
    raw = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    raw_rows = len(raw)
    missing_customer_ids = int(raw["CustomerID"].isna().sum())
    duplicate_rows = int(raw.duplicated().sum())
    zero_price_rows = int(pd.to_numeric(raw["UnitPrice"], errors="coerce").eq(0).sum())
    negative_price_rows = int(pd.to_numeric(raw["UnitPrice"], errors="coerce").lt(0).sum())

    df = raw.copy()
    for col in ["InvoiceNo", "StockCode", "Description", "Country"]:
        df[col] = df[col].astype("string").str.strip()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ["Quantity", "UnitPrice", "CustomerID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates().copy()

    # Fill missing descriptions only when the StockCode has an observed description.
    description_map = (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .first()
    )
    df["Description"] = df["Description"].fillna(df["StockCode"].map(description_map))

    cancel_mask = df["InvoiceNo"].str.upper().str.startswith("C", na=False)
    debt_mask = df["Description"].str.contains("bad debt", case=False, na=False)

    cancellations = df[cancel_mask].copy()
    bad_debt = df[debt_mask].copy()
    returns = df[(~cancel_mask) & (~debt_mask) & (df["Quantity"] < 0)].copy()

    # Sales dashboard uses ordinary positive-quantity, positive-price line items.
    sales_mask = (
        (~cancel_mask)
        & (~debt_mask)
        & df["InvoiceDate"].notna()
        & df["Quantity"].gt(0)
        & df["UnitPrice"].gt(0)
        & df["InvoiceNo"].notna()
        & df["StockCode"].notna()
    )
    sales = df[sales_mask].copy()
    sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
    sales = sales.sort_values("InvoiceDate")

    customer_sales = sales.dropna(subset=["CustomerID"]).copy()
    customer_sales["CustomerID"] = customer_sales["CustomerID"].astype("int64")
    customer_sales = customer_sales.sort_values("InvoiceDate")

    # Full-period RFM values for customer segmentation.
    reference_date = customer_sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = customer_sales.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max().normalize()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        Country=("Country", "last")
    ).reset_index()

    # K-Means: same four-cluster setup as the notebook.
    rfm_features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[rfm_features])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    # Quintile RFM scores as in the notebook.
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"), 5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), 5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), 5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    def segment_customer(row):
        if row["R_Score"] >= 4 and row["F_Score"] >= 4:
            return "Loyal"
        elif row["M_Score"] >= 4:
            return "High-Value"
        elif row["R_Score"] <= 2:
            return "Inactive"
        return "Occasional"

    rfm["Segment"] = rfm.apply(segment_customer, axis=1)

    # --------------------------------------------------------
    # Repeat-purchase model follows the structure in the notebook:
    # 80% chronological cutoff, historical RFM features, next-30-day label,
    # then a stratified customer-level validation split.
    # --------------------------------------------------------
    min_date = customer_sales["InvoiceDate"].min()
    max_date = customer_sales["InvoiceDate"].max()
    split_date = min_date + (max_date - min_date) * 0.80

    history = customer_sales[customer_sales["InvoiceDate"] < split_date].copy()
    future = customer_sales[
        (customer_sales["InvoiceDate"] >= split_date)
        & (customer_sales["InvoiceDate"] < split_date + pd.Timedelta(days=30))
    ].copy()

    future_customers = set(future["CustomerID"].unique())

    rfm_train = history.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (split_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        Country=("Country", "last")
    )
    rfm_train["Target"] = rfm_train.index.isin(future_customers).astype(int)

    # Historical RFM labels for the prediction snapshot (no use of later-period behaviour).
    rfm_train["R_Score"] = pd.qcut(
        rfm_train["Recency"].rank(method="first"), 5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int)
    rfm_train["F_Score"] = pd.qcut(
        rfm_train["Frequency"].rank(method="first"), 5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm_train["M_Score"] = pd.qcut(
        rfm_train["Monetary"].rank(method="first"), 5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm_train["RFM_Score"] = (
        rfm_train["R_Score"].astype(str)
        + rfm_train["F_Score"].astype(str)
        + rfm_train["M_Score"].astype(str)
    )

    def historical_segment(row):
        if row["R_Score"] >= 4 and row["F_Score"] >= 4:
            return "Loyal"
        elif row["M_Score"] >= 4:
            return "High-Value"
        elif row["R_Score"] <= 2:
            return "Inactive"
        return "Occasional"

    rfm_train["Segment"] = rfm_train.apply(historical_segment, axis=1)

    model_results = {}
    prediction_table = pd.DataFrame()
    model_metrics = pd.DataFrame()
    confusion_tables = {}

    features = ["Recency", "Frequency", "Monetary"]
    X = rfm_train[features]
    y = rfm_train["Target"]

    if len(rfm_train) >= 30 and y.nunique() == 2 and y.value_counts().min() >= 2:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Logistic Regression uses scaled values.
        lr_scaler = StandardScaler()
        X_train_scaled = lr_scaler.fit_transform(X_train)
        X_val_scaled = lr_scaler.transform(X_val)

        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_val_scaled)
        lr_prob = lr_model.predict_proba(X_val_scaled)[:, 1]

        # Random Forest uses the original RFM feature scale.
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_val)
        rf_prob = rf_model.predict_proba(X_val)[:, 1]

        def metric_row(name, y_true, pred, prob):
            return {
                "Model": name,
                "Accuracy": accuracy_score(y_true, pred),
                "Precision": precision_score(y_true, pred, zero_division=0),
                "Recall": recall_score(y_true, pred, zero_division=0),
                "F1 Score": f1_score(y_true, pred, zero_division=0),
                "ROC AUC": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan
            }

        model_metrics = pd.DataFrame([
            metric_row("Logistic Regression", y_val, lr_pred, lr_prob),
            metric_row("Random Forest", y_val, rf_pred, rf_prob)
        ])

        validation = X_val.copy()
        validation["ActualRepurchase"] = y_val
        validation["LR_Probability"] = lr_prob
        validation["LR_Prediction"] = lr_pred
        validation["RF_Probability"] = rf_prob
        validation["RF_Prediction"] = rf_pred
        validation["CustomerID"] = validation.index.astype(int)
        validation = validation.reset_index(drop=True)

        meta = rfm_train[["Country", "Segment", "RFM_Score"]].copy()
        meta["CustomerID"] = meta.index.astype(int)
        meta = meta.reset_index(drop=True)

        prediction_table = validation.merge(meta, on="CustomerID", how="left")
        prediction_table["ActualLabel"] = prediction_table["ActualRepurchase"].map(
            {0: "No repurchase", 1: "Repurchased"}
        )

        confusion_tables["Logistic Regression"] = confusion_matrix(y_val, lr_pred)
        confusion_tables["Random Forest"] = confusion_matrix(y_val, rf_pred)
        model_results = {
            "Logistic Regression": {"probability_col": "LR_Probability", "prediction_col": "LR_Prediction"},
            "Random Forest": {"probability_col": "RF_Probability", "prediction_col": "RF_Prediction"}
        }

    metadata = {
        "raw_rows": raw_rows,
        "duplicate_rows": duplicate_rows,
        "missing_customer_ids": missing_customer_ids,
        "zero_price_rows": zero_price_rows,
        "negative_price_rows": negative_price_rows,
        "cancel_rows": len(cancellations),
        "bad_debt_rows": len(bad_debt),
        "return_rows": len(returns),
        "sales_rows": len(sales),
        "identified_customers": int(customer_sales["CustomerID"].nunique()),
        "split_date": split_date,
        "future_label_end": split_date + pd.Timedelta(days=30),
        "target_rate": float(rfm_train["Target"].mean()) if len(rfm_train) else np.nan,
        "target_customers": int(rfm_train["Target"].sum()) if len(rfm_train) else 0,
        "model_snapshot_customers": len(rfm_train)
    }

    return sales, rfm, prediction_table, model_metrics, confusion_tables, metadata


# ------------------------------------------------------------
# Load workbook
# ------------------------------------------------------------
st.sidebar.header("Data and filters")
uploaded_file = st.sidebar.file_uploader(
    "Upload Online Retail.xlsx",
    type=["xlsx"]
)

if uploaded_file is not None:
    workbook_bytes = uploaded_file.getvalue()
else:
    workbook_path = Path(__file__).with_name("Online Retail.xlsx")
    if not workbook_path.exists():
        st.info(
            "Upload the original **Online Retail.xlsx** workbook using the sidebar. "
            "Keep it in the same folder as app.py or upload it here."
        )
        st.stop()
    workbook_bytes = workbook_path.read_bytes()

try:
    sales, rfm, predictions, model_metrics, confusion_tables, metadata = prepare_dashboard(workbook_bytes)
except Exception as exc:
    st.error(f"Could not prepare the dashboard from this workbook: {exc}")
    st.stop()

if sales.empty:
    st.error("No positive-quantity, positive-price sales were found after cleaning.")
    st.stop()


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
countries = sorted(sales["Country"].dropna().astype(str).unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country",
    options=countries,
    default=countries
)

min_day = sales["InvoiceDate"].min().date()
max_day = sales["InvoiceDate"].max().date()
date_range = st.sidebar.date_input(
    "Invoice date range",
    value=(min_day, max_day),
    min_value=min_day,
    max_value=max_day
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_day, end_day = date_range
else:
    start_day, end_day = min_day, max_day

filtered_sales = sales[
    sales["Country"].astype(str).isin(selected_countries)
    & (sales["InvoiceDate"] >= pd.Timestamp(start_day))
    & (sales["InvoiceDate"] < pd.Timestamp(end_day) + pd.Timedelta(days=1))
].copy()

filtered_rfm = rfm[rfm["Country"].astype(str).isin(selected_countries)].copy()
filtered_predictions = predictions[
    predictions["Country"].astype(str).isin(selected_countries)
].copy() if not predictions.empty else predictions.copy()

st.sidebar.caption(
    "Sales charts follow the selected date range. RFM and prediction results use "
    "the historical customer snapshots described in the dashboard."
)


# ------------------------------------------------------------
# KPI row
# ------------------------------------------------------------
total_revenue = filtered_sales["Revenue"].sum()
total_orders = filtered_sales["InvoiceNo"].nunique()
total_customers = filtered_sales["CustomerID"].nunique()
avg_order_value = total_revenue / total_orders if total_orders else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sales revenue", f"£{total_revenue:,.0f}")
k2.metric("Orders", f"{total_orders:,}")
k3.metric("Identified customers", f"{total_customers:,}")
k4.metric("Average order value", f"£{avg_order_value:,.2f}")

st.caption(
    "Revenue is calculated as Quantity × UnitPrice for positive-quantity, positive-price "
    "sales lines. Cancellation invoices, bad-debt adjustments, returns, and non-positive-price "
    "lines are not included in this sales view."
)


# ------------------------------------------------------------
# Dashboard tabs
# ------------------------------------------------------------
sales_tab, customer_tab, prediction_tab, data_tab = st.tabs([
    "Sales Overview",
    "Customer Segments",
    "Repeat-Purchase Prediction",
    "Data Quality"
])

with sales_tab:
    st.subheader("Sales trends")
    if filtered_sales.empty:
        st.warning("No sales match the selected country and date filters.")
    else:
        monthly = (
            filtered_sales.assign(Month=filtered_sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
            .groupby("Month", as_index=False)["Revenue"].sum()
        )
        fig = px.line(
            monthly, x="Month", y="Revenue", markers=True,
            title="Monthly sales revenue", template="plotly_white"
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            product_summary = (
                filtered_sales.groupby("StockCode")
                .agg(
                    Description=("Description", "first"),
                    QuantitySold=("Quantity", "sum"),
                    Revenue=("Revenue", "sum")
                )
                .nlargest(10, "QuantitySold")
                .reset_index()
            )
            product_summary["DisplayProduct"] = product_summary["Description"].fillna(
                product_summary["StockCode"]
            )
            fig_products = px.bar(
                product_summary.sort_values("QuantitySold"),
                x="QuantitySold", y="DisplayProduct", orientation="h",
                hover_data=["StockCode", "Revenue"],
                title="Top 10 products by quantity sold",
                template="plotly_white"
            )
            fig_products.update_layout(xaxis_title="Quantity sold", yaxis_title="Product")
            st.plotly_chart(fig_products, use_container_width=True)

        with right:
            country_summary = (
                filtered_sales.groupby("Country", as_index=False)["Revenue"]
                .sum().nlargest(10, "Revenue")
            )
            fig_country = px.bar(
                country_summary.sort_values("Revenue"),
                x="Revenue", y="Country", orientation="h",
                title="Top 10 countries by revenue", template="plotly_white"
            )
            fig_country.update_layout(xaxis_title="Revenue (£)", yaxis_title="Country")
            st.plotly_chart(fig_country, use_container_width=True)

        st.subheader("Purchase frequency")
        frequency = filtered_sales.dropna(subset=["CustomerID"]).groupby("CustomerID")["InvoiceNo"].nunique()
        if not frequency.empty:
            frequency_groups = pd.cut(
                frequency,
                bins=[0, 1, 2, 5, 10, float("inf")],
                labels=["1 purchase", "2 purchases", "3–5", "6–10", "10+"]
            )
            frequency_counts = frequency_groups.value_counts().sort_index().reset_index()
            frequency_counts.columns = ["Purchase frequency", "Customers"]
            fig_frequency = px.bar(
                frequency_counts, x="Purchase frequency", y="Customers",
                title="Customers by purchase-frequency group",
                template="plotly_white"
            )
            st.plotly_chart(fig_frequency, use_container_width=True)

with customer_tab:
    st.subheader("RFM customer segments")
    if filtered_rfm.empty:
        st.warning("No identified customers match the selected countries.")
    else:
        left, right = st.columns([1, 1])
        with left:
            segment_counts = (
                filtered_rfm["Segment"].value_counts()
                .rename_axis("Segment").reset_index(name="Customers")
            )
            fig_segments = px.bar(
                segment_counts, x="Segment", y="Customers", color="Segment",
                title="Customers by RFM segment", template="plotly_white"
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        with right:
            cluster_profile = filtered_rfm.groupby("Cluster")[
                ["Recency", "Frequency", "Monetary"]
            ].mean()
            cluster_profile = (cluster_profile - cluster_profile.mean()) / cluster_profile.std(ddof=0).replace(0, 1)
            fig_cluster = px.imshow(
                cluster_profile,
                text_auto=".1f", aspect="auto", color_continuous_scale="RdBu_r",
                title="Standardized RFM profile by K-Means cluster"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

        st.subheader("Customer RFM table")
        display_rfm = filtered_rfm[[
            "CustomerID", "Recency", "Frequency", "Monetary",
            "RFM_Score", "Cluster", "Segment", "Country"
        ]].sort_values("Monetary", ascending=False)
        st.dataframe(
            display_rfm,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Monetary": st.column_config.NumberColumn("Monetary (£)", format="£%.2f"),
                "Recency": st.column_config.NumberColumn("Recency (days)")
            }
        )
        st.download_button(
            "Download filtered RFM customers",
            data=display_rfm.to_csv(index=False).encode("utf-8"),
            file_name="rfm_customers.csv",
            mime="text/csv"
        )

with prediction_tab:
    st.subheader("30-day repeat-purchase probabilities")
    if predictions.empty or model_metrics.empty:
        st.warning(
            "The notebook-style prediction model could not be trained. "
            "Check that the dataset contains enough identified customers and both target classes."
        )
    else:
        st.info(
            "These are probabilities for a held-out validation sample at one historical cutoff. "
            "They demonstrate the model workflow; they are not verified forecasts for customers "
            "after the final date in the workbook."
        )

        st.markdown("**Model comparison on the validation customers**")
        st.dataframe(
            model_metrics.style.format({
                "Accuracy": "{:.1%}", "Precision": "{:.1%}",
                "Recall": "{:.1%}", "F1 Score": "{:.3f}", "ROC AUC": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        selected_model = st.selectbox(
            "Choose model probabilities",
            list(model_metrics["Model"]),
            index=(list(model_metrics["Model"]).index("Random Forest")
                   if "Random Forest" in list(model_metrics["Model"]) else 0)
        )
        probability_col = "RF_Probability" if selected_model == "Random Forest" else "LR_Probability"
        prediction_col = "RF_Prediction" if selected_model == "Random Forest" else "LR_Prediction"

        scored = filtered_predictions.copy()
        if scored.empty:
            st.warning("No validation customers match the selected countries.")
        else:
            scored["PredictedProbability"] = scored[probability_col]
            threshold = st.slider(
                "Highlight customers above this probability",
                min_value=0.0, max_value=1.0, value=0.50, step=0.05
            )

            high_probability = int((scored["PredictedProbability"] >= threshold).sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Validation customers", f"{len(scored):,}")
            m2.metric("Above selected threshold", f"{high_probability:,}")
            m3.metric("Observed 30-day repurchase rate", f"{scored['ActualRepurchase'].mean():.1%}")

            fig_prob = px.histogram(
                scored, x="PredictedProbability", nbins=20,
                color="ActualLabel", barmode="overlay", opacity=0.75,
                title=f"{selected_model}: predicted probability distribution",
                template="plotly_white"
            )
            fig_prob.update_layout(xaxis_title="Predicted probability", yaxis_title="Customers")
            st.plotly_chart(fig_prob, use_container_width=True)

            def recommend(row):
                p = row["PredictedProbability"]
                segment = row.get("Segment", "Occasional")
                recency = row["Recency"]
                monetary_score = row["Monetary"]
                if p >= 0.75:
                    return "High likelihood: loyalty reward or relevant cross-sell"
                if segment == "High-Value" and (p < 0.50 or recency > 90):
                    return "Protect valuable customer: personalized win-back offer"
                if segment == "Inactive":
                    return "Reactivation: low-cost reminder or targeted offer"
                if p >= 0.40:
                    return "Potential repeat buyer: personalized product recommendation"
                if monetary_score > 0:
                    return "Low likelihood: monitor and test a modest re-engagement message"
                return "Monitor customer activity"

            scored["Recommendation"] = scored.apply(recommend, axis=1)
            scored["PredictedClass"] = scored[prediction_col].map(
                {0: "No repurchase", 1: "Repurchase"}
            )
            scored["ActualOutcome"] = scored["ActualLabel"]

            st.subheader("Customers and suggested actions")
            show_cols = [
                "CustomerID", "Country", "Segment", "RFM_Score",
                "PredictedProbability", "PredictedClass", "ActualOutcome",
                "Recommendation"
            ]
            display_predictions = scored[show_cols].sort_values(
                "PredictedProbability", ascending=False
            )
            st.dataframe(
                display_predictions,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "PredictedProbability": st.column_config.ProgressColumn(
                        "Purchase probability", min_value=0, max_value=1, format="%.2f"
                    )
                }
            )
            st.download_button(
                "Download validation predictions",
                data=display_predictions.to_csv(index=False).encode("utf-8"),
                file_name="repeat_purchase_validation_predictions.csv",
                mime="text/csv"
            )

            cm = confusion_tables.get(selected_model)
            if cm is not None:
                st.subheader("Confusion matrix at 0.50 threshold")
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    x=["Predicted: No", "Predicted: Yes"],
                    y=["Actual: No", "Actual: Yes"],
                    color_continuous_scale="Blues",
                    aspect="auto"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

with data_tab:
    st.subheader("Data quality and modelling notes")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Original rows", f"{metadata['raw_rows']:,}")
    q2.metric("Exact duplicates removed", f"{metadata['duplicate_rows']:,}")
    q3.metric("Missing customer IDs", f"{metadata['missing_customer_ids']:,}")
    q4.metric("Clean positive sales rows", f"{metadata['sales_rows']:,}")

    st.markdown("**Excluded or separately identified transaction rows**")
    quality_table = pd.DataFrame({
        "Record type": [
            "Cancellation invoices", "Bad-debt adjustments",
            "Negative-quantity records / returns", "Zero UnitPrice", "Negative UnitPrice"
        ],
        "Rows": [
            metadata["cancel_rows"], metadata["bad_debt_rows"],
            metadata["return_rows"], metadata["zero_price_rows"], metadata["negative_price_rows"]
        ]
    })
    st.dataframe(quality_table, hide_index=True, use_container_width=True)

    st.markdown("**Predictive modelling snapshot**")
    st.write(f"Historical cutoff: **{metadata['split_date']:%d %b %Y %H:%M}**")
    st.write(f"30-day label window ends: **{metadata['future_label_end']:%d %b %Y %H:%M}**")
    st.write(f"Customers with historical feature records: **{metadata['model_snapshot_customers']:,}**")
    st.write(f"Repurchase rate in this labelled snapshot: **{metadata['target_rate']:.1%}**")

    st.warning(
        "The notebook creates one historical customer snapshot and then randomly splits those "
        "customers into train and validation sets. This is a useful project baseline, but it is "
        "not a full walk-forward time-series evaluation. Also, the dashboard excludes returns and "
        "non-positive prices from sales revenue; these records are counted separately."
    )

st.divider()
st.caption(
    "Online Retail project dashboard · Revenue shown in GBP (£) · "
    "Customer-level analysis excludes records without CustomerID."
)
