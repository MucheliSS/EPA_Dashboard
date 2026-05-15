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
