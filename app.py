import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessor, helper

st.set_page_config(page_title=" WhatsApp Analyzer ", layout="wide")

st.title(" WhatsApp Chat Analyzer ")

uploaded_file = st.sidebar.file_uploader("Upload Chat File")

if uploaded_file is not None:
    data = uploaded_file.getvalue().decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Select User", user_list)

    if st.sidebar.button("Analyze"):

        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🤖 AI", "📈 Advanced"])

        #  TAB 1
        with tab1:
            st.header("📊 Stats")

            num_messages, words, media, links = helper.fetch_stats(selected_user, df)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Messages", num_messages)
            col2.metric("Words", words)
            col3.metric("Media", media)
            col4.metric("Links", links)

            st.subheader("☁️ WordCloud")
            wc = helper.create_wordcloud(df)
            st.image(wc.to_array())

            st.subheader("😂 Emoji Pie Chart")
            emoji_df = helper.emoji_helper(df)

            if not emoji_df.empty:
                fig, ax = plt.subplots()
                ax.pie(emoji_df['count'], labels=emoji_df['emoji'], autopct="%0.2f%%")
                st.pyplot(fig)

            st.subheader("🔥 Common Words")
            common_df = helper.most_common_words(selected_user, df)

            fig, ax = plt.subplots()
            ax.barh(common_df['word'], common_df['count'])
            st.pyplot(fig)

        #  TAB 2
        with tab2:
            st.header("🤖 Sentiment Analysis")

            df = helper.sentiment_analysis(df)
            sentiment = helper.sentiment_counts(df)

            fig, ax = plt.subplots()
            ax.pie(sentiment.values, labels=sentiment.index, autopct="%0.2f%%")
            st.pyplot(fig)

            st.bar_chart(sentiment)

            st.write(f"💡 Most messages are **{sentiment.idxmax()}**")

        # TAB 3
        with tab3:
            st.header("📈 Advanced Insights")

            st.subheader("📅 Monthly Timeline")
            timeline = helper.monthly_timeline(df)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'])
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.subheader("⏱️ Hourly Activity")
            hourly = helper.hourly_activity(df)

            fig, ax = plt.subplots()
            ax.bar(hourly.index, hourly.values)
            st.pyplot(fig)

            st.subheader("🔥 Heatmap")
            heatmap = helper.activity_heatmap(df)

            fig, ax = plt.subplots()
            sns.heatmap(heatmap)
            st.pyplot(fig)

            st.subheader("🔗 Top Links")
            for link, count in helper.top_links(df):
                st.write(f"{link} → {count}")