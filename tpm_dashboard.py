# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import calendar
import re
from datetime import date, datetime
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
        s.astype("string")
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

def pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


FEEDBACK_MONTH_LABEL_COL = "__Feedback Month__"
FEEDBACK_MONTH_SORT_COL = "__Feedback Month Sort__"
UNKNOWN_MONTH = "Unknown month"

MONTH_COLUMN_CANDIDATES = [
    "Feedback Month",
    "Month of Feedback",
    "Assessment Month",
    "EPA Month",
    "Rotation Month",
    "Month",
    "Feedback Date",
    "Date of Feedback",
    "Assessment Date",
    "Date of Assessment",
    "Submission Date",
    "Date Submitted",
    "Submitted Date",
    "Created Date",
    "Timestamp",
    "Date",
]

REMARKS_COLUMN_CANDIDATES = [
    "Remarks",
    "Remark",
    "Comments",
    "Comment",
    "Feedback",
    "Qualitative Feedback",
    "Narrative Feedback",
    "Additional Comments",
    "Assessor Comments",
    "Evaluator Comments",
]


def _column_tokens(name: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(name).lower())


def _norm_column_name(name: str) -> str:
    return " ".join(_column_tokens(name))


def pick_feedback_month_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None

    normalized = {_norm_column_name(c): c for c in df.columns}
    for candidate in MONTH_COLUMN_CANDIDATES:
        match = normalized.get(_norm_column_name(candidate))
        if match is not None:
            return match

    for c in df.columns:
        tokens = set(_column_tokens(c))
        if "month" in tokens:
            return c

    for c in df.columns:
        tokens = set(_column_tokens(c))
        if tokens.intersection({"date", "timestamp"}):
            return c

    return None


def pick_remarks_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None

    normalized = {_norm_column_name(c): c for c in df.columns}
    for candidate in REMARKS_COLUMN_CANDIDATES:
        match = normalized.get(_norm_column_name(candidate))
        if match is not None:
            return match

    for c in df.columns:
        tokens = set(_column_tokens(c))
        is_comment = tokens.intersection({"remark", "remarks", "comment", "comments", "feedback"})
        is_metadata = tokens.intersection({"date", "month", "time", "timestamp"})
        if is_comment and not is_metadata:
            return c

    return None


def _numeric_month(value) -> Optional[int]:
    if isinstance(value, bool) or pd.isna(value):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number.is_integer() and 1 <= int(number) <= 12:
        return int(number)
    return None


def _month_period(value):
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return pd.Timestamp(value.year, value.month, 1)

    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value.year, value.month, 1)

    if _numeric_month(value) is not None:
        return pd.NaT

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return pd.NaT

    if _numeric_month(text) is not None:
        return pd.NaT

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT

    return pd.Timestamp(parsed.year, parsed.month, 1)


def format_feedback_month(value) -> str:
    period = _month_period(value)
    if pd.notna(period):
        return period.strftime("%b %Y")

    month_number = _numeric_month(value)
    if month_number is not None:
        return calendar.month_abbr[month_number]

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return UNKNOWN_MONTH

    return text


def feedback_month_sort_value(value):
    period = _month_period(value)
    if pd.notna(period):
        return period

    month_number = _numeric_month(value)
    if month_number is not None:
        return pd.Timestamp(1900, month_number, 1)

    return pd.NaT


def add_feedback_month_columns(df: pd.DataFrame, source_col: Optional[str]) -> pd.DataFrame:
    if source_col and source_col in df.columns:
        df[FEEDBACK_MONTH_LABEL_COL] = df[source_col].apply(format_feedback_month)
        df[FEEDBACK_MONTH_SORT_COL] = df[source_col].apply(feedback_month_sort_value)
    return df


def ordered_feedback_months(df: pd.DataFrame) -> List[str]:
    if FEEDBACK_MONTH_LABEL_COL not in df.columns:
        return []

    columns = [FEEDBACK_MONTH_LABEL_COL]
    if FEEDBACK_MONTH_SORT_COL in df.columns:
        columns.append(FEEDBACK_MONTH_SORT_COL)

    months = df[columns].drop_duplicates(subset=[FEEDBACK_MONTH_LABEL_COL]).copy()
    months["__input_order__"] = range(len(months))
    months["__unknown__"] = months[FEEDBACK_MONTH_LABEL_COL].eq(UNKNOWN_MONTH)

    if FEEDBACK_MONTH_SORT_COL in months.columns:
        months["__missing_sort__"] = months[FEEDBACK_MONTH_SORT_COL].isna()
        months = months.sort_values(
            ["__unknown__", "__missing_sort__", FEEDBACK_MONTH_SORT_COL, "__input_order__"],
            na_position="last",
        )
    else:
        months = months.sort_values(["__unknown__", "__input_order__"])

    return months[FEEDBACK_MONTH_LABEL_COL].astype(str).tolist()


def clean_score_columns(df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if score_columns:
        df[score_columns] = df[score_columns].mask(df[score_columns].eq(0))
    return df

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
    quant_month_col = pick_feedback_month_column(df_quant)
    df_quant = add_feedback_month_columns(df_quant, quant_month_col)

    # Coerce numeric scores and treat 0/blank values as missing for averages.
    df_quant = clean_score_columns(df_quant, domains_present)

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
    qual_month_col = None
    if not df_qual.empty:
        qual_month_col = pick_feedback_month_column(df_qual)
        df_qual = add_feedback_month_columns(df_qual, qual_month_col)
        if "Resident Name" in df_qual.columns:
            df_qual["Resident Name"] = _norm_str_series(df_qual["Resident Name"])
        remarks_col = pick_remarks_column(df_qual)
        if remarks_col:
            df_qual[remarks_col] = _norm_str_series(df_qual[remarks_col])

    st.success("✅ Data loaded successfully! GM scores normalized (÷2); 0 and blank scores excluded from averages.")

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
                df_res_all = df_quant[df_quant["Resident Name"] == resident].copy()

                score_month_options = ordered_feedback_months(df_res_all)
                selected_score_month = "All months"
                if score_month_options:
                    selected_score_month = st.selectbox(
                        "Choose Feedback Month for Scores",
                        ["All months"] + score_month_options,
                        key="score_month",
                    )

                if selected_score_month != "All months":
                    df_res = df_res_all[
                        df_res_all[FEEDBACK_MONTH_LABEL_COL] == selected_score_month
                    ].copy()
                else:
                    df_res = df_res_all.copy()

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
                elif df_res.empty:
                    st.warning("No quantitative scores found for this resident and month selection.")
                else:
                    month_id_vars = []
                    labels = {evaluator_col: "Assessor"}
                    hover_data = None
                    if FEEDBACK_MONTH_LABEL_COL in df_res.columns:
                        month_id_vars.append(FEEDBACK_MONTH_LABEL_COL)
                        labels[FEEDBACK_MONTH_LABEL_COL] = "Feedback Month"
                        hover_data = {FEEDBACK_MONTH_LABEL_COL: True}

                    df_melt = df_res.melt(
                        id_vars=[evaluator_col] + month_id_vars,
                        value_vars=domain_line,
                        var_name="Domain",
                        value_name="Score",
                    ).dropna(subset=["Score"])

                    st.write(f"**Domain scores for {resident} by assessor:**")
                    if df_melt.empty:
                        st.warning("No plottable scores found for this selection.")
                    else:
                        line_kwargs = {
                            "x": "Domain",
                            "y": "Score",
                            "color": evaluator_col,
                            "markers": True,
                            "title": f"EPA Domain Scores - {resident}",
                            "labels": labels,
                        }
                        if hover_data:
                            line_kwargs["hover_data"] = hover_data

                        fig = px.line(df_melt, **line_kwargs)
                        fig.update_layout(yaxis=dict(range=[0, 5]))
                        st.plotly_chart(fig, use_container_width=True)

                    if score_month_options:
                        trend_id_vars = [FEEDBACK_MONTH_LABEL_COL]
                        if FEEDBACK_MONTH_SORT_COL in df_res_all.columns:
                            trend_id_vars.append(FEEDBACK_MONTH_SORT_COL)

                        df_trend = df_res_all.melt(
                            id_vars=trend_id_vars,
                            value_vars=domain_line,
                            var_name="Domain",
                            value_name="Score",
                        ).dropna(subset=["Score"])

                        if not df_trend.empty:
                            month_order = ordered_feedback_months(df_res_all)
                            monthly = (
                                df_trend.groupby([FEEDBACK_MONTH_LABEL_COL, "Domain"], as_index=False)["Score"]
                                .mean()
                            )
                            monthly["__month_order__"] = monthly[FEEDBACK_MONTH_LABEL_COL].map(
                                {month: idx for idx, month in enumerate(month_order)}
                            )
                            monthly = monthly.sort_values(["__month_order__", "Domain"])

                            st.write("### Feedback Month Trend")
                            fig_trend = px.line(
                                monthly,
                                x=FEEDBACK_MONTH_LABEL_COL,
                                y="Score",
                                color="Domain",
                                markers=True,
                                title=f"Average Domain Scores by Feedback Month - {resident}",
                                labels={
                                    FEEDBACK_MONTH_LABEL_COL: "Feedback Month",
                                    "Score": "Average Score",
                                },
                                category_orders={FEEDBACK_MONTH_LABEL_COL: month_order},
                            )
                            fig_trend.update_layout(yaxis=dict(range=[0, 5]))
                            st.plotly_chart(fig_trend, use_container_width=True)

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
                remarks_col = pick_remarks_column(df_qual)

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

                selected_comment_month = "All months"
                if FEEDBACK_MONTH_LABEL_COL in resident_qual.columns:
                    comment_month_options = ordered_feedback_months(resident_qual)
                    if comment_month_options:
                        selected_comment_month = st.selectbox(
                            "Choose Feedback Month for Comments",
                            ["All months"] + comment_month_options,
                            key="comments_month",
                        )
                        if selected_comment_month != "All months":
                            resident_qual = resident_qual[
                                resident_qual[FEEDBACK_MONTH_LABEL_COL] == selected_comment_month
                            ].copy()

                # Relax the filter: any non-empty text
                meaningful = resident_qual[
                    resident_qual[remarks_col].notna() &
                    (resident_qual[remarks_col].str.strip().str.len() > 0)
                ].copy()

                caption_parts = [
                    f"Qual rows: {len(df_qual)}",
                    f"Filtered for '{selected_resident}': {len(resident_qual)}",
                    f"With text in '{remarks_col}': {len(meaningful)}",
                ]
                if selected_comment_month != "All months":
                    caption_parts.insert(2, f"Filtered for month '{selected_comment_month}'")
                st.caption(" | ".join(caption_parts))

                if meaningful.empty:
                    st.write("No comments found for this selection.")
                else:
                    view = st.radio("View comments as:", ["Raw Comments", "Sentiment Analysis", "Word Cloud"])
                    comments_list = meaningful[remarks_col].astype(str).tolist()

                    if view == "Sentiment Analysis":
                        analysis_label = str(selected_resident)
                        if selected_comment_month != "All months":
                            analysis_label = f"{analysis_label} - {selected_comment_month}"
                        display_sentiment_analysis(analysis_label, comments_list)

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
                            label_parts = []
                            if FEEDBACK_MONTH_LABEL_COL in row and pd.notna(row[FEEDBACK_MONTH_LABEL_COL]):
                                month_label = str(row[FEEDBACK_MONTH_LABEL_COL]).strip()
                                if month_label and month_label != UNKNOWN_MONTH:
                                    label_parts.append(month_label)
                            if selected_resident == "All" and resident_col_q and pd.notna(row[resident_col_q]):
                                label_parts.append(str(row[resident_col_q]))
                            label_parts.append(who)
                            with st.expander(f"📝 {' | '.join(label_parts)}"):
                                if FEEDBACK_MONTH_LABEL_COL in row and pd.notna(row[FEEDBACK_MONTH_LABEL_COL]):
                                    st.caption(f"Feedback month: {row[FEEDBACK_MONTH_LABEL_COL]}")
                                st.write(row[remarks_col])
else:
    st.info("Please upload an EPA Excel file to begin.")

st.markdown("---")
st.caption("🔢 GM scores are automatically normalized (÷2) for fair comparison with EPA scores.")
