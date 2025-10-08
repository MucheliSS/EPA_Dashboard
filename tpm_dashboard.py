# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Optional

# Optional libs: guard-import so the app still works without them
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except Exception:
    WordCloud, plt = None, None

# -----------------------------
# Utilities
# -----------------------------
def _norm_str_series(s: pd.Series) -> pd.Series:
    """Trim extra whitespace and normalize to string."""
    return (
        s.astype(str)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

def pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -----------------------------
# Sentiment helpers (safe)
# -----------------------------
def analyze_sentiment_textblob(comments: List[str]) -> pd.DataFrame:
    if TextBlob is None:
        return pd.DataFrame(columns=["comment", "sentiment", "polarity_score", "confidence"])

    rows = []
    for c in comments:
        if pd.isna(c):
            continue
        s = str(c).strip()
        if not s:
            continue

        blob = TextBlob(s)
        polarity = float(blob.sentiment.polarity)

        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        trunc = s if len(s) <= 150 else s[:150] + "..."
        rows.append(
            {"comment": trunc, "sentiment": label, "polarity_score": polarity, "confidence": abs(polarity)}
        )
    return pd.DataFrame(rows)


def create_sentiment_summary(df: pd.DataFrame):
    if df.empty or "sentiment" not in df:
        return None
    counts = df["sentiment"].value_counts().to_dict()
    for k in ("Positive", "Neutral", "Negative"):
        counts.setdefault(k, 0)
    total = int(df.shape[0])
    pct = {k: (counts[k] / total * 100.0 if total else 0.0) for k in ("Positive", "Neutral", "Negative")}
    return {"counts": counts, "percentages": pct, "total": total}


def display_sentiment_analysis(resident_name: str, comments: List[str]) -> None:
    st.subheader(f"📊 Sentiment Analysis for {resident_name}")

    if not comments:
        st.warning("No comments available for sentiment analysis.")
        return

    df = analyze_sentiment_textblob(comments)
    if df.empty:
        if TextBlob is None:
            st.info("TextBlob not installed — run `pip install textblob` to enable sentiment analysis.")
        else:
            st.warning("No valid comments to analyze.")
        return

    summary = create_sentiment_summary(df)
    if not summary:
        st.warning("Could not compute summary.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Comments", summary["total"])
    with c2:
        st.metric("Positive", f"{summary['counts']['Positive']} ({summary['percentages']['Positive']:.1f}%)")
    with c3:
        st.metric("Neutral", f"{summary['counts']['Neutral']} ({summary['percentages']['Neutral']:.1f}%)")
    with c4:
        st.metric("Negative", f"{summary['counts']['Negative']} ({summary['percentages']['Negative']:.1f}%)")

    # Pie (remove invalid 'color' arg; rely on color_discrete_map)
    fig = px.pie(
        values=[summary["counts"]["Positive"], summary["counts"]["Neutral"], summary["counts"]["Negative"]],
        names=["Positive", "Neutral", "Negative"],
        title="Sentiment Distribution",
        color_discrete_map={"Positive": "#2E8B57", "Neutral": "#FFD700", "Negative": "#DC143C"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.subheader("📈 Key Insights")
    pos = summary["percentages"]["Positive"]
    neg = summary["percentages"]["Negative"]
    if pos > 60:
        st.success(f"✅ Mostly positive feedback ({pos:.1f}% positive).")
    elif neg > 40:
        st.error(f"⚠️ Concerning feedback patterns ({neg:.1f}% negative).")
    else:
        st.info("ℹ️ Mixed feedback — review individual comments for context.")

    # Extremes (guard idxmax/min if single row)
    if "polarity_score" in df and not df.empty:
        most_pos = df.loc[df["polarity_score"].idxmax()]
        most_neg = df.loc[df["polarity_score"].idxmin()]
        st.write("**Most Positive Comment:**")
        st.success(most_pos["comment"])
        st.write("**Most Critical Comment:**")
        st.error(most_neg["comment"])

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="EPA Dashboard", layout="wide")
st.title("EPA Assessment Dashboard")

uploaded_file = st.file_uploader("Upload your EPA Excel file", type=["xlsx"])

if uploaded_file:
    # Load Quantitative
    try:
        df_quant = pd.read_excel(uploaded_file, sheet_name="Quantitative")
    except Exception as e:
        st.error(f"Could not load Quantitative sheet: {e}")
        st.stop()

    # Normalize key strings early
    if "Resident Name" in df_quant.columns:
        df_quant["Resident Name"] = _norm_str_series(df_quant["Resident Name"])

    # Normalize GM scores if Assessment Type exists
    if "Assessment Type" in df_quant.columns:
        is_gm = df_quant["Assessment Type"].astype(str).str.contains("GM", case=False, na=False)
    else:
        is_gm = pd.Series(False, index=df_quant.index)

    domains_all = ["PC", "MK", "SBP", "PBLI", "Prof", "ICS", "Overall"]
    domains_present = [c for c in domains_all if c in df_quant.columns]

    # Coerce numeric
    for col in domains_present:
        df_quant[col] = pd.to_numeric(df_quant[col], errors="coerce")

    # Divide GM by 2 only where present
    if is_gm.any() and len(domains_present) > 0:
        df_quant.loc[is_gm, domains_present] = df_quant.loc[is_gm, domains_present] / 2.0

    # Load Qualitative (optional)
    df_qual = pd.DataFrame()
    try:
        uploaded_file.seek(0)  # rewind for 2nd read
        df_qual = pd.read_excel(uploaded_file, sheet_name="Qualitative")
    except Exception as e:
        st.warning(f"Qualitative sheet not available: {e}")

    # Normalize Qual strings
    remarks_col = None
    if not df_qual.empty:
        if "Resident Name" in df_qual.columns:
            df_qual["Resident Name"] = _norm_str_series(df_qual["Resident Name"])
        for c in df_qual.columns:
            if "remark" in c.lower() or "comment" in c.lower():
                remarks_col = c
                break
        if remarks_col:
            df_qual[remarks_col] = _norm_str_series(df_qual[remarks_col])

    st.success("✅ Data loaded successfully! GM scores normalized (÷2) where applicable.")

    tab1, tab2, tab3 = st.tabs(["📊 Individual SR", "💬 Comments & Sentiment", "🏆 Overall Ranking"])

    with tab1:
        if "Resident Name" not in df_quant.columns:
            st.error("`Resident Name` column not found in Quantitative sheet.")
        else:
            residents = df_quant["Resident Name"].dropna().unique().tolist()
            if not residents:
                st.warning("No residents found.")
            else:
                resident = st.selectbox("Choose Resident", residents)
                df_res = df_quant[df_quant["Resident Name"] == resident].copy()

                # Pick evaluator column
                evaluator_col = pick_first_present(df_res, ["Name of Evaluator", "Assessor", "Evaluator"])
                if evaluator_col is None:
                    st.info("No evaluator column found; using row index as evaluator.")
                    df_res["__Evaluator__"] = [f"Eval {i+1}" for i in range(len(df_res))]
                    evaluator_col = "__Evaluator__"

                # Which domains available (exclude Overall for line chart)
                domain_line = [c for c in ["PC", "MK", "SBP", "PBLI", "Prof", "ICS"] if c in df_res.columns]
                if not domain_line:
                    st.warning("No domain columns found to plot.")
                else:
                    df_melt = df_res.melt(
                        id_vars=[evaluator_col],
                        value_vars=domain_line,
                        var_name="Domain",
                        value_name="Score",
                    ).dropna(subset=["Score"])

                    st.write(f"**Domain scores for {resident} by assessor:**")
                    fig = px.line(
                        df_melt,
                        x="Domain",
                        y="Score",
                        color=evaluator_col,
                        markers=True,
                        title=f"EPA Domain Scores - {resident}",
                    )
                    fig.update_layout(yaxis=dict(range=[0, 5]))
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("### 📊 Average Domain Scores")
                    avgs = df_res[domain_line].mean(numeric_only=True)
                    avg_table = pd.DataFrame(avgs).T
                    avg_table.index = ["Average"]
                    st.dataframe(avg_table.style.format("{:.2f}"), use_container_width=True, height=70)

    with tab3:
        st.subheader("🏆 Overall Ranking of Senior Residents")
        if "Resident Name" in df_quant.columns and "Overall" in df_quant.columns:
            resident_averages = (
                df_quant.groupby("Resident Name")[domains_present].mean(numeric_only=True).round(2)
            )
            if not resident_averages.empty:
                resident_averages = resident_averages.sort_values("Overall", ascending=False)
                resident_averages["Rank"] = range(1, len(resident_averages) + 1)
                ordered_cols = ["Rank"]
                if "Overall" in resident_averages.columns:
                    ordered_cols += ["Overall"]
                ordered_cols += [c for c in ["PC", "MK", "SBP", "PBLI", "Prof", "ICS"] if c in resident_averages.columns]
                resident_averages = resident_averages[ordered_cols]
                st.dataframe(resident_averages, use_container_width=True)

                st.write("### 📊 Overall Score Comparison")
                fig_ranking = px.bar(
                    x=resident_averages.index,
                    y=resident_averages["Overall"],
                    title="Average Overall EPA Scores by Resident",
                    labels={"x": "Resident", "y": "Average Overall Score"},
                )
                fig_ranking.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_ranking, use_container_width=True)
            else:
                st.info("No averages available to rank.")
        else:
            st.info("`Resident Name` or `Overall` column missing; cannot compute ranking.")

    with tab2:
        if df_qual.empty:
            st.write("No qualitative data sheet available.")
        else:
            # Identify columns
            resident_col_q = "Resident Name" if "Resident Name" in df_qual.columns else None
            if remarks_col is None:
                # Detect on the fly if not set
                for c in df_qual.columns:
                    if "remark" in c.lower() or "comment" in c.lower():
                        remarks_col = c
                        break

            if not remarks_col:
                st.write("No remarks/comments column found in qualitative data.")
            else:
                # Build resident choices from UNION of Quant + Qual
                residents_quant = (
                    df_quant["Resident Name"].dropna().unique().tolist()
                    if "Resident Name" in df_quant.columns else []
                )
                residents_qual = (
                    df_qual["Resident Name"].dropna().unique().tolist()
                    if resident_col_q else []
                )
                resident_choices = sorted(set(residents_quant) | set(residents_qual))
                resident_choices = ["All"] + resident_choices if resident_choices else ["All"]

                selected_resident = st.selectbox(
                    "Choose Resident for Comments",
                    resident_choices,
                    key="comments_resident",
                )

                # Filter by resident if possible
                if resident_col_q and selected_resident != "All":
                    resident_qual = df_qual[df_qual[resident_col_q].str.casefold() ==
                                            str(selected_resident).casefold()].copy()
                else:
                    resident_qual = df_qual.copy()
                    if not resident_col_q:
                        st.info("No resident column in qualitative data. Showing all comments.")

                # Relax the filter: any non-empty text
                meaningful = resident_qual[
                    resident_qual[remarks_col].notna() &
                    (resident_qual[remarks_col].str.strip().str.len() > 0)
                ].copy()

                # Debug counts so you can see data flowing
                st.caption(
                    f"Qual rows: {len(df_qual)} | "
                    f"Filtered for '{selected_resident}': {len(resident_qual)} | "
                    f"With text in '{remarks_col}': {len(meaningful)}"
                )

                if meaningful.empty:
                    st.write("No comments found for this selection.")
                else:
                    view = st.radio("View comments as:", ["Raw Comments", "Sentiment Analysis", "Word Cloud"])
                    comments_list = meaningful[remarks_col].astype(str).tolist()

                    if view == "Sentiment Analysis":
                        display_sentiment_analysis(str(selected_resident), comments_list)

                    elif view == "Word Cloud":
                        if WordCloud is None or plt is None:
                            st.info("WordCloud not installed — run `pip install wordcloud matplotlib` to enable.")
                        else:
                            all_text = " ".join(comments_list).strip()
                            if len(all_text) < 10:
                                st.write("Not enough comment data for word cloud.")
                            else:
                                wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(all_text)
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wc, interpolation="bilinear")
                                ax.axis("off")
                                st.pyplot(fig)
                    else:
                        st.subheader("📝 All Comments")
                        assessor_col = pick_first_present(meaningful, ["Assessor", "Name of Evaluator", "Evaluator"])
                        for i, row in meaningful.reset_index(drop=True).iterrows():
                            who = str(row[assessor_col]) if assessor_col and pd.notna(row[assessor_col]) else f"Assessor {i+1}"
                            with st.expander(f"📝 {who}"):
                                st.write(row[remarks_col])
else:
    st.info("Please upload an EPA Excel file to begin.")

st.markdown("---")
st.caption("🔢 GM scores are automatically normalized (÷2) for fair comparison with EPA scores.")
