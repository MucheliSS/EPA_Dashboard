import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="EPA Dashboard", layout="wide")
st.title("EPA Assessment Dashboard")

uploaded_file = st.file_uploader("Upload your EPA Excel file", type=["xlsx"])

if uploaded_file:
    # Load quantitative data
    df = pd.read_excel(uploaded_file, sheet_name="Quantitative")
    df['is_GM'] = df['Assessment Type'].astype(str).str.contains('GM')
    for col in ['PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS', 'Overall']:
        df.loc[df['is_GM'], col] = pd.to_numeric(df.loc[df['is_GM'], col], errors='coerce') / 2.0
        df.loc[~df['is_GM'], col] = pd.to_numeric(df.loc[~df['is_GM'], col], errors='coerce')

    # Try to load qualitative data from individual sheets
    qualitative_data = []
    try:
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        for sheet in sheet_names:
            if sheet != 'Quantitative' and sheet != 'Sheet5' and sheet != 'Qualitative':
                sheet_df = pd.read_excel(uploaded_file, sheet_name=sheet)
                # Extract comments from 'C. ADDITIONAL COMMENTS' column
                if 'C. ADDITIONAL COMMENTS' in sheet_df.columns:
                    comments = sheet_df['C. ADDITIONAL COMMENTS'].dropna()
                    for comment in comments:
                        if str(comment).strip() and len(str(comment)) > 10:  # Filter meaningful comments
                            qualitative_data.append({
                                'Resident Name': sheet,
                                'Comments': str(comment),
                                'Source': 'Individual Sheet'
                            })
    except:
        st.warning("Could not load qualitative data - continuing with quantitative only.")

    df_comments = pd.DataFrame(qualitative_data) if qualitative_data else pd.DataFrame()

    st.success("Data loaded and normalized! GM scores have been divided by 2 for parity.")

    # Resident selection
    residents = df['Resident Name'].dropna().unique()
    resident = st.selectbox("Choose Resident", residents)
    df_res = df[df['Resident Name'] == resident]

    # Create tabs
    tab1, tab2 = st.tabs(["📊 Scores & Charts", "💬 Comments"])
    
    with tab1:
        # Your existing score visualization code
        domains = ['PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS']
        df_melt = df_res.melt(
            id_vars=['Name of Evaluator'],
            value_vars=domains,
            var_name='Domain', value_name='Score'
        )
        df_melt = df_melt.dropna(subset=['Score'])

        st.write(f"**Domain-by-domain scores for {resident}, by assessor:**")

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

        # Average scores
        avgs = df_res[domains].mean()
        st.write("### Average Domain Scores")
        avg_table = pd.DataFrame(avgs).T
        avg_table.index = ['Mean']
        st.dataframe(avg_table.style.format("{:.2f}"), height=70)
    
    with tab2:
        # Comments section
        if not df_comments.empty and resident in df_comments['Resident Name'].values:
            st.write(f"### Comments for {resident}")
            
            # Display options
            comment_view = st.radio("View comments as:", ["Word Cloud", "Expandable List", "Full Table"])
            
            resident_comments = df_comments[df_comments['Resident Name'] == resident]['Comments']
            
            if comment_view == "Word Cloud":
                all_text = ' '.join(resident_comments.astype(str))
                if len(all_text.strip()) > 20:
                    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(all_text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.write("Not enough comment data for word cloud.")
            
            elif comment_view == "Expandable List":
                for i, comment in enumerate(resident_comments, 1):
                    with st.expander(f"Comment {i}"):
                        st.write(comment)
            
            else:  # Full Table
                st.dataframe(df_comments[df_comments['Resident Name'] == resident], use_container_width=True)
        
        else:
            st.write(f"No qualitative comments found for {resident}.")

else:
    st.info("Please upload an EPA Excel file to begin.")

st.markdown("---")
st.caption("GM scores are automatically normalized to the EPA scale (divided by 2).")
