import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EPA Dashboard", layout="wide")

st.title("EPA Assessment Dashboard")

uploaded_file = st.file_uploader("Upload your EPA Excel file", type=["xlsx"])

if uploaded_file:
    # Load the quantitative data
    df = pd.read_excel(uploaded_file, sheet_name="Quantitative")
    df['is_GM'] = df['Assessment Type'].astype(str).str.contains('GM')
    # Normalize GM scores
    for col in ['PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS', 'Overall']:
        df.loc[df['is_GM'], col] = pd.to_numeric(df.loc[df['is_GM'], col], errors='coerce') / 2.0
        df.loc[~df['is_GM'], col] = pd.to_numeric(df.loc[~df['is_GM'], col], errors='coerce')
    
    st.success("Data loaded and normalized! GM scores have been divided by 2 for parity.")

    # Interactive resident selection
    residents = df['Resident Name'].dropna().unique()
    resident = st.selectbox("Choose Resident", residents)
    df_res = df[df['Resident Name'] == resident]

    st.write(f"**All Assessments for {resident}:**")
    st.dataframe(df_res)

    # Show trend over time for this resident (overall score)
    st.write("**Trend of Overall EPA Score**")
    fig = px.line(df_res, x="Month of Assessment", y="Overall", color="Assessment Type", markers=True, title="Overall Score Trend")
    st.plotly_chart(fig, use_container_width=True)

    # Show per-domain chart for most recent year
    st.write("**Domain Scores (most recent year)**")
    recent = df_res[df_res['Year of Assessment'] == df_res['Year of Assessment'].max()]
    domains = ['PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS']
    domain_scores = recent[domains].melt(var_name='Domain', value_name='Score')
    fig2 = px.bar(domain_scores, x='Domain', y='Score', color='Domain', title='Domain Scores')
    st.plotly_chart(fig2, use_container_width=True)

    # Show averages for all residents
    st.write("**Resident Average Overall Scores** (GM normalized)")
    avg_overall = df.groupby('Resident Name')['Overall'].mean().sort_values(ascending=False)
    st.dataframe(avg_overall.reset_index().rename(columns={"Overall": "Average Overall"}))
    fig3 = px.bar(avg_overall, x=avg_overall.index, y='Overall', title="Average Overall Score by Resident")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Please upload an EPA Excel file to begin.")

st.markdown("---")
st.caption("GM scores are automatically normalized to the EPA scale (divided by 2).")
