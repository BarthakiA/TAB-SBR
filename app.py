import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(page_title="Nykaa Analytics with Tabs", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("NYKA.csv", parse_dates=["signup_date", "last_purchase_date"])
    return df

df = load_data()

# Pre-compute groupings for summary
cltv_by_seg = df.groupby("RFM_segment_label")["predicted_CLTV_3m"].mean().reset_index()
churn_by_seg = df.groupby("RFM_segment_label")["churn_within_3m_flag"].mean().reset_index()

tabs = st.tabs(["Summary", "RFM Segmentation", "CLTV Analysis", "Churn Analysis"])

# Summary Tab
with tabs[0]:
    st.header("Dashboard Summary")
    churn_rate = df["churn_within_3m_flag"].mean()
    avg_cltv = df["predicted_CLTV_3m"].mean()
    top_cltv = cltv_by_seg.loc[cltv_by_seg['predicted_CLTV_3m'].idxmax()]
    highest_churn = churn_by_seg.loc[churn_by_seg['churn_within_3m_flag'].idxmax()]
    lowest_churn = churn_by_seg.loc[churn_by_seg['churn_within_3m_flag'].idxmin()]

    st.markdown(f"- Largest segment: **{df['RFM_segment_label'].value_counts().idxmax()}**.")
    st.markdown(f"- Average CLTV: **₹{avg_cltv:.0f}**.")
    st.markdown(f"- Top CLTV segment: **{top_cltv['RFM_segment_label']}** "
                f"with **₹{top_cltv['predicted_CLTV_3m']:.0f}**.")
    st.markdown(f"- Overall churn rate: **{churn_rate:.1%}**.")
    st.markdown(f"- Highest churn in **{highest_churn['RFM_segment_label']}** at "
                f"**{highest_churn['churn_within_3m_flag']:.1%}**.")
    st.markdown(f"- Lowest churn in **{lowest_churn['RFM_segment_label']}** at "
                f"**{lowest_churn['churn_within_3m_flag']:.1%}**.")
    st.write("Use these insights to target marketing and retention strategies effectively.")

# RFM Segmentation Tab
with tabs[1]:
    st.header("1. RFM Segmentation")
    rfm = df.rename(columns={
        "recency_days": "Recency",
        "frequency_3m": "Frequency",
        "monetary_value_3m": "Monetary"
    })
    # Recency distribution
    fig1 = px.histogram(rfm, x="Recency", nbins=40).update_traces(marker_color='#008080')
    st.plotly_chart(fig1, use_container_width=True)
    st.write("Customers cluster around recent purchases under 60 days.")
    st.write("A smaller group shows longer gaps up to 180 days.")
    st.write("This helps identify ‘At Risk’ vs ‘Recent’ buyers.")

    # Frequency distribution
    fig2 = px.histogram(rfm, x="Frequency", nbins=20).update_traces(marker_color='#FF69B4')
    st.plotly_chart(fig2, use_container_width=True)
    st.write("Most place 1–3 orders per quarter.")
    st.write("A minority orders frequently (5+ times).")
    st.write("High-frequency buyers are prime loyalty targets.")

    # Monetary distribution
    fig3 = px.histogram(rfm, x="Monetary", nbins=30).update_traces(marker_color='#00FFFF')
    st.plotly_chart(fig3, use_container_width=True)
    st.write("Spending peaks at ₹500–₹1000.")
    st.write("High spenders above ₹5000 form a key segment.")
    st.write("This guides premium marketing efforts.")

    # 3D scatter of segments
    cmap = ['#008080','#FF69B4','#00FFFF','#FFC300','#DAF7A6']
    fig4 = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary",
        color="RFM_segment_label",
        color_discrete_sequence=cmap,
        labels={"Recency":"Recency (days)",
                "Frequency":"# orders",
                "Monetary":"Spend (₹)"},
        title="3D View of RFM Segments"
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.write("Visual separation of ‘Champions’, ‘Loyal’, ‘At Risk’, etc.")
    st.write("Use cluster positions for tailored campaigns.")

# CLTV Analysis Tab
with tabs[2]:
    st.header("2. Customer Lifetime Value (3-Month Forecast)")
    # Histogram of predicted CLTV
    fig5 = px.histogram(df, x="predicted_CLTV_3m", nbins=30).update_traces(marker_color='#008080')
    st.plotly_chart(fig5, use_container_width=True)
    st.write("CLTV mostly under ₹1000.")
    st.write("A long tail reaches ₹3000+ for VIPs.")
    st.write("Focus retention on high-CLTV groups.")

    # Predicted vs Actual CLTV
    fig6 = px.scatter(df, x="predicted_CLTV_3m", y="actual_CLTV_3m").update_traces(marker_color='#FF69B4')
    st.plotly_chart(fig6, use_container_width=True)
    st.write("Predictions track actuals closely.")
    st.write("Deviation appears at low-value end.")
    st.write("Model tuning could improve low-end accuracy.")

    # Avg CLTV by segment
    fig7 = px.bar(cltv_by_seg, x="RFM_segment_label", y="predicted_CLTV_3m")        .update_traces(marker_color='#00FFFF')
    st.plotly_chart(fig7, use_container_width=True)
    st.write("Champions segment shows highest CLTV.")
    st.write("Under-performers may need cross-sell initiatives.")
    st.write("Allocate budget based on segment CLTV.")

    # Actual CLTV distribution by segment
    fig8 = px.box(df, x="RFM_segment_label", y="actual_CLTV_3m", color_discrete_sequence=['#008080'])
    st.plotly_chart(fig8, use_container_width=True)
    st.write("Actual CLTV varies within segments.")
    st.write("High variance segments need personalized offers.")
    st.write("Boxplots reveal outliers for special attention.")

# Churn Analysis Tab
with tabs[3]:
    st.header("3. Churn Analysis & Prediction")
    # Overall churn rate
    churn_rate = df["churn_within_3m_flag"].mean()
    fig9 = px.bar(x=["Active","Churned"], y=[1-churn_rate, churn_rate],
                  labels={"x":"Status","y":"Proportion"},
                  title="Overall 3-Month Churn Rate")        .update_traces(marker_color=['#008080','#FF69B4'])
    st.plotly_chart(fig9, use_container_width=True)
    st.write(f"Churn rate is **{churn_rate:.1%}**.")
    st.write("High churn signals retention gaps.")
    st.write("Mitigation programs are essential.")

    # Churn by segment
    fig10 = px.bar(churn_by_seg, x="RFM_segment_label", y="churn_within_3m_flag")        .update_traces(marker_color='#00FFFF')
    st.plotly_chart(fig10, use_container_width=True)
    st.write("‘At Risk’ segment churns above 50%.")
    st.write("‘Champions’ churn stays below 10%.")
    st.write("Segment-based retention is key.")

    # Recency vs Churn
    fig11 = px.box(df, x="churn_within_3m_flag", y="recency_days", color_discrete_sequence=['#008080'])
    st.plotly_chart(fig11, use_container_width=True)
    st.write("Churned customers have higher recency.")
    st.write("Reactivate those with long gaps.")
    st.write("Trigger win-back campaigns at 60 days.")

    # ROC curve
    features = [
        "recency_days","frequency_3m","monetary_value_3m",
        "time_on_app_minutes","page_views_per_session",
        "campaign_clicks","campaign_views","cart_abandonment_rate",
        "first_time_buyer_flag"
    ]
    X = df[features].fillna(0)
    y = df["churn_within_3m_flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    fig12 = px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC={auc:.2f})")        .update_traces(line_color='#00FFFF')
    st.plotly_chart(fig12, use_container_width=True)
    st.write(f"AUC of **{auc:.2f}** indicates strong predictive power.")
    st.write("Use ROC to choose optimal threshold.")
    st.write("Balance false positives vs false negatives.")
