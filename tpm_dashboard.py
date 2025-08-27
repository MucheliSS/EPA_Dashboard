import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="EPA Dashboard", layout="wide")

st.title("EPA Assessment Dashboard")

uploaded_file = st.file_uploader("Upload your EPA Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Quantitative")
    df['is_GM'] = df['Assessment Type'].astype(str).str.contains('GM')
    for col in ['PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS', 'Overall']:
        df.loc[df['is_GM'], col] = pd.to_numeric(df.loc[df['is_GM'], col], errors='coerce') / 2.0
        df.loc[~df['is_GM'], col] = pd.to_numeric(df.loc[~df['is_GM'], col], errors='coerce')

    st.success("Data loaded and normalized! GM scores have been divided by 2 for parity.")

    # Resident selection
    residents = df['Resident Name'].dropna().unique()
    resident = st.selectbox("Choose Resident", residents)
    df_res = df[df['Resident Name'] == resident]

    # Assessor/Domain line plot
    domains = ['PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS']
    df_melt = df_res.melt(
        id_vars=['Name of Evaluator'],
        value_vars=domains,
        var_name='Domain', value_name='Score'
    )
    df_melt = df_melt.dropna(subset=['Score'])

    st.write(f"**Domain-by-domain scores for {resident}, by assessor:**")

    # Line plot: Each assessor is a line
    fig = px.line(
        df_melt,
        x='Domain',
        y='Score',
        color='Name of Evaluator',
        markers=True,
        line_dash='Name of Evaluator'
    )
    fig.update_layout(yaxis=dict(range=[0, 5]), title=None)
    st.plotly_chart(fig, use_container_width=True)

    # Show average scores per domain as numbers
    avgs = df_res[domains].mean()
    st.write("### Average Domain Scores")
    # Display as a horizontal number table
    avg_table = pd.DataFrame(avgs).T
    avg_table.index = ['Mean']
    st.dataframe(avg_table.style.format("{:.2f}"), height=70)

else:
    st.info("Please upload an EPA Excel file to begin.")

st.markdown("---")
st.caption("GM scores are automatically normalized to the EPA scale (divided by 2).")

