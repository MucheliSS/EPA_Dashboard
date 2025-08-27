import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sentiment Analysis Functions
def analyze_sentiment_textblob(comments):
    """Analyze sentiment using TextBlob"""
    results = []
    for comment in comments:
        if pd.isna(comment) or str(comment).strip() == '':
            continue
        
        blob = TextBlob(str(comment))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        results.append({
            'comment': str(comment)[:150] + '...' if len(str(comment)) > 150 else str(comment),
            'sentiment': sentiment,
            'polarity_score': polarity,
            'confidence': abs(polarity)
        })
    
    return pd.DataFrame(results)

def create_sentiment_summary(sentiment_df):
    """Create summary statistics from sentiment analysis"""
    if sentiment_df.empty:
        return None
    
    summary = sentiment_df['sentiment'].value_counts()
    total = len(sentiment_df)
    
    # Ensure all categories are present
    for category in ['Positive', 'Neutral', 'Negative']:
        if category not in summary:
            summary[category] = 0
    
    percentages = {
        'Positive': (summary.get('Positive', 0) / total) * 100,
        'Neutral': (summary.get('Neutral', 0) / total) * 100,
        'Negative': (summary.get('Negative', 0) / total) * 100
    }
    
    return {
        'counts': summary.to_dict(),
        'percentages': percentages,
        'total': total
    }

def display_sentiment_analysis(resident_name, comments):
    """Display comprehensive sentiment analysis in Streamlit"""
    st.subheader(f"📊 Sentiment Analysis for {resident_name}")
    
    if not comments or len(comments) == 0:
        st.warning("No comments available for sentiment analysis.")
        return
    
    # Perform analysis
    sentiment_df = analyze_sentiment_textblob(comments)
    
    if sentiment_df.empty:
        st.warning("No valid comments to analyze.")
        return
    
    # Create summary
    summary = create_sentiment_summary(sentiment_df)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Comments", summary['total'])
    with col2:
        st.metric("Positive", f"{summary['counts']['Positive']} ({summary['percentages']['Positive']:.1f}%)")
    with col3:
        st.metric("Neutral", f"{summary['counts']['Neutral']} ({summary['percentages']['Neutral']:.1f}%)")
    with col4:
        st.metric("Negative", f"{summary['counts']['Negative']} ({summary['percentages']['Negative']:.1f}%)")
    
    # Visualization
    fig = px.pie(values=list(summary['counts'].values()), 
                 names=list(summary['counts'].keys()),
                 title="Sentiment Distribution",
                 color_discrete_map={
                     'Positive': '#2E8B57',
                     'Neutral': '#FFD700', 
                     'Negative': '#DC143C'
                 })
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("📈 Key Insights")
    
    if summary['percentages']['Positive'] > 60:
        st.success(f"✅ **Mostly positive feedback** ({summary['percentages']['Positive']:.1f}% positive)")
    elif summary['percentages']['Negative'] > 40:
        st.error(f"⚠️ **Concerning feedback patterns** ({summary['percentages']['Negative']:.1f}% negative)")
    else:
        st.info(f"ℹ️ **Mixed feedback** - Review individual comments for context")
    
    # Most extreme comments
    if 'polarity_score' in sentiment_df.columns:
        most_positive = sentiment_df.nlargest(1, 'polarity_score')
        most_negative = sentiment_df.nsmallest(1, 'polarity_score')
        
        if not most_positive.empty:
            st.write("**Most Positive Comment:**")
            st.success(most_positive.iloc[0]['comment'])
        
        if not most_negative.empty:
            st.write("**Most Critical Comment:**")
            st.error(most_negative.iloc[0]['comment'])

# Main Dashboard
st.set_page_config(page_title="EPA Dashboard", layout="wide")
st.title("EPA Assessment Dashboard")

uploaded_file = st.file_uploader("Upload your EPA Excel file", type=["xlsx"])

if uploaded_file:
    # Load quantitative data
    df_quant = pd.read_excel(uploaded_file, sheet_name="Quantitative")
    df_quant['is_GM'] = df_quant['Assessment Type'].astype(str).str.contains('GM')
    
    # Normalize GM scores
    for col in ['PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS', 'Overall']:
        df_quant.loc[df_quant['is_GM'], col] = pd.to_numeric(df_quant.loc[df_quant['is_GM'], col], errors='coerce') / 2.0
        df_quant.loc[~df_quant['is_GM'], col] = pd.to_numeric(df_quant.loc[~df_quant['is_GM'], col], errors='coerce')

    # Load qualitative data
    try:
        df_qual = pd.read_excel(uploaded_file, sheet_name="Qualitative")
        st.success("✅ Data loaded successfully! GM scores normalized (÷2) for parity.")
    except Exception as e:
        df_qual = pd.DataFrame()
        st.warning(f"Could not load Qualitative sheet: {e}")
        st.success("✅ Quantitative data loaded! GM scores normalized (÷2) for parity.")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Individual SR", "💬 Comments & Sentiment" ,"🏆 Overall Ranking"])
    
    with tab1:
        # Individual SR Analysis
        residents = df_quant['Resident Name'].dropna().unique()
        resident = st.selectbox("Choose Resident", residents)
        df_res = df_quant[df_quant['Resident Name'] == resident]

        # Domain scores by assessor (line plot)
        domains = ['PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS']
        df_melt = df_res.melt(
            id_vars=['Name of Evaluator'],
            value_vars=domains,
            var_name='Domain', 
            value_name='Score'
        )
        df_melt = df_melt.dropna(subset=['Score'])

        st.write(f"**Domain scores for {resident} by assessor:**")

        # Line plot: Each assessor is a different colored line
        fig = px.line(
            df_melt,
            x='Domain',
            y='Score',
            color='Name of Evaluator',
            markers=True,
            title=f"EPA Domain Scores - {resident}"
        )
        fig.update_layout(yaxis=dict(range=[0, 5]))
        st.plotly_chart(fig, use_container_width=True)

        # Average scores as numbers
        st.write("### 📊 Average Domain Scores")
        avgs = df_res[domains].mean()
        avg_table = pd.DataFrame(avgs).T
        avg_table.index = ['Average']
        st.dataframe(avg_table.style.format("{:.2f}"), use_container_width=True, height=70)
    
    with tab3:
        # Overall Ranking of All SRs
        st.subheader("🏆 Overall Ranking of Senior Residents")
        
        # Calculate average scores for each resident
        resident_averages = df_quant.groupby('Resident Name')[['PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS', 'Overall']].mean().round(2)
        
        # Sort by Overall score (descending)
        resident_averages = resident_averages.sort_values('Overall', ascending=False)
        
        # Add rank column
        resident_averages['Rank'] = range(1, len(resident_averages) + 1)
        
        # Reorder columns to show Rank first
        resident_averages = resident_averages[['Rank', 'Overall', 'PC', 'MK', 'SBP', 'PBLI', 'Prof', 'ICS']]
        
        # Display as interactive table
        st.dataframe(resident_averages, use_container_width=True)
        
        # Bar chart of overall scores
        st.write("### 📊 Overall Score Comparison")
        fig_ranking = px.bar(
            x=resident_averages.index,
            y=resident_averages['Overall'],
            title="Average Overall EPA Scores by Resident",
            labels={'x': 'Resident', 'y': 'Average Overall Score'}
        )
        fig_ranking.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_ranking, use_container_width=True)
    
    with tab2:
        # Comments and sentiment analysis
        residents = df_quant['Resident Name'].dropna().unique()
        selected_resident = st.selectbox("Choose Resident for Comments", residents, key="comments_resident")
        
        if not df_qual.empty:
            # Filter qualitative data for this resident
            if 'Resident Name' in df_qual.columns:
                resident_qual = df_qual[df_qual['Resident Name'] == selected_resident]
            else:
                # If no resident name column, show all comments
                resident_qual = df_qual
                st.info("No resident-specific filtering available in qualitative data. Showing all comments.")
            
            if not resident_qual.empty:
                st.write(f"### Comments for {selected_resident}")
                
                # Look for remarks column (default, no dropdown)
                remarks_columns = []
                for col in resident_qual.columns:
                    if 'remark' in col.lower() or 'comment' in col.lower():
                        remarks_columns.append(col)
                
                if remarks_columns:
                    # Use first remarks column found
                    remarks_column = remarks_columns[0]
                    
                    # Get comments and assessor info
                    if 'Assessor' in resident_qual.columns:
                        assessor_col = 'Assessor'
                    elif 'Name of Evaluator' in resident_qual.columns:
                        assessor_col = 'Name of Evaluator'
                    elif 'Evaluator' in resident_qual.columns:
                        assessor_col = 'Evaluator'
                    else:
                        assessor_col = None
                    
                    # Filter meaningful comments
                    meaningful_comments = resident_qual[
                        (resident_qual[remarks_column].notna()) & 
                        (resident_qual[remarks_column].astype(str).str.len() > 10)
                    ]
                    
                    if not meaningful_comments.empty:
                        comments_list = meaningful_comments[remarks_column].astype(str).tolist()
                        
                        # Display options
                        comment_view = st.radio("View comments as:", 
                                               ["Sentiment Analysis", "Word Cloud", "Raw Comments"])
                        
                        if comment_view == "Sentiment Analysis":
                            display_sentiment_analysis(selected_resident, comments_list)
                        
                        elif comment_view == "Word Cloud":
                            all_text = ' '.join(comments_list)
                            if len(all_text.strip()) > 50:
                                wordcloud = WordCloud(width=800, height=400, 
                                                    background_color='white', 
                                                    max_words=100).generate(all_text)
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                            else:
                                st.write("Not enough comment data for word cloud.")
                        
                        else:  # Raw Comments
                            st.subheader("📝 All Comments")
                            for idx, row in meaningful_comments.iterrows():
                                # Use assessor name if available, otherwise generic
                                if assessor_col and pd.notna(row[assessor_col]):
                                    assessor_name = str(row[assessor_col])
                                else:
                                    assessor_name = f"Assessor {idx + 1}"
                                
                                with st.expander(f"📝 {assessor_name}"):
                                    st.write(row[remarks_column])
                    else:
                        st.write("No meaningful remarks found for this resident.")
                else:
                    st.write("No remarks column found in qualitative data.")
            else:
                st.write(f"No qualitative data found for {selected_resident}.")
        else:
            st.write("No qualitative data sheet available.")

else:
    st.info("Please upload an EPA Excel file to begin.")

st.markdown("---")
st.caption("🔢 GM scores are automatically normalized (÷2) for fair comparison with EPA scores.")

