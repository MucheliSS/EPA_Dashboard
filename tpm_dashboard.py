import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

    # Resident selection
    residents = df_quant['Resident Name'].dropna().unique()
    resident = st.selectbox("Choose Resident", residents)
    df_res = df_quant[df_quant['Resident Name'] == resident]

    # Create tabs
    tab1, tab2 = st.tabs(["📊 Scores & Charts", "💬 Comments & Sentiment"])
    
    with tab1:
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
    
    with tab2:
        # Comments and sentiment analysis
        if not df_qual.empty:
            # Filter qualitative data for this resident
            if 'Resident Name' in df_qual.columns:
                resident_qual = df_qual[df_qual['Resident Name'] == resident]
            else:
                # If no resident name column, show all comments
                resident_qual = df_qual
                st.info("No resident-specific filtering available in qualitative data. Showing all comments.")
            
            if not resident_qual.empty:
                st.write(f"### Comments for {resident}")
                
                # Find comment columns (look for columns with text data)
                comment_columns = []
                for col in resident_qual.columns:
                    if resident_qual[col].dtype == 'object' and col.lower() != 'resident name':
                        comment_columns.append(col)
                
                if comment_columns:
                    # Let user choose which comment field to analyze
                    selected_column = st.selectbox("Select comment field:", comment_columns)
                    
                    # Get comments from selected column
                    comments = resident_qual[selected_column].dropna().astype(str).tolist()
                    # Filter out very short comments
                    comments = [c for c in comments if len(c.strip()) > 10]
                    
                    if comments:
                        # Display options
                        comment_view = st.radio("View comments as:", 
                                               ["Sentiment Analysis", "Word Cloud", "Raw Comments"])
                        
                        if comment_view == "Sentiment Analysis":
                            display_sentiment_analysis(resident, comments)
                        
                        elif comment_view == "Word Cloud":
                            all_text = ' '.join(comments)
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
                            for i, comment in enumerate(comments, 1):
                                with st.expander(f"Comment {i}"):
                                    st.write(comment)
                    else:
                        st.write("No meaningful comments found for this resident.")
                else:
                    st.write("No comment columns found in qualitative data.")
            else:
                st.write(f"No qualitative data found for {resident}.")
        else:
            st.write("No qualitative data sheet available.")

else:
    st.info("Please upload an EPA Excel file to begin.")

st.markdown("---")
st.caption("🔢 GM scores are automatically normalized (÷2) for fair comparison with EPA scores.")
