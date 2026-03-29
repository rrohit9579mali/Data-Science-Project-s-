import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 🔥 Page Config
st.set_page_config(page_title="WhatsApp Analyzer", page_icon="💬", layout="wide")

# 🔥 Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #00FFAB;
}
.metric-box {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# 🔥 Header
st.markdown('<div class="title">💬 WhatsApp Chat Analyzer</div>', unsafe_allow_html=True)
st.write("Analyze your chats like a pro 🚀")

# 🔥 Sidebar
st.sidebar.title("📂 Control Panel")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    with st.spinner("Analyzing chats... ⏳"):
        data = uploaded_file.getvalue().decode("utf-8")
        df = preprocessor.preprocess(data)

    st.success("✅ Analysis Ready!")

    # --- Sidebar selections (Taaki page refresh na ho) ---
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Select User", user_list)

    years = df['year'].unique().tolist()
    years.sort()
    selected_year = st.sidebar.selectbox("Select Year for Activity Map", years)

    # Interest Analysis ke liye person selection
    interest_users = [u for u in user_list if u != "Overall"]
    person_name = st.sidebar.selectbox("Select Person for Interest Analysis", interest_users)

    # 🔥 SINGLE BUTTON TRIGGER
    if st.sidebar.button("Show Full Analysis"):

        sns.set_style("darkgrid")
        plt.style.use("dark_background")

        # 🔥 Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.markdown("## 📊 Top Statistics")
        col1, col2, col3, col4 = st.columns(4)


        def card(title, value):
            return f'<div class="metric-box"><h3>{title}</h3><h1>{value}</h1></div>'


        with col1:
            st.markdown(card("Messages", num_messages), unsafe_allow_html=True)
        with col2:
            st.markdown(card("Words", words), unsafe_allow_html=True)
        with col3:
            st.markdown(card("Media", num_media_messages), unsafe_allow_html=True)
        with col4:
            st.markdown(card("Links", num_links), unsafe_allow_html=True)

        # 📅 Monthly Timeline
        st.markdown("## 📅 Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], marker='o', color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # 📆 Daily Timeline
        st.markdown("## 📆 Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='orange')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # 📊 Activity Map
        st.markdown("## 📊 Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header(f"Most busy month in {selected_year}")
            # Year filtering logic (As it is)
            year_df = df[df['year'] == selected_year]
            busy_month = helper.month_activity_map(selected_user, year_df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # 🔥 Heatmap
        st.markdown("## 🔥 Weekly Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(user_heatmap, ax=ax)
        st.pyplot(fig)

        # 👥 Most Busy Users (Group Level)
        if selected_user == 'Overall':
            st.markdown("## 👥Most Time Spent With")
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # ☁ Wordcloud
        st.markdown("## ☁ Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        ax.axis('off')
        st.pyplot(fig)

        # 🔤 Most Common Words
        st.markdown("## 🔤 Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        st.pyplot(fig)

        # 😀 Emoji Analysis
        st.markdown("## 😀 Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            if not emoji_df.empty:
                fig = px.pie(emoji_df.head(), values=1, names=0, hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

        # ❤️ INTEREST ANALYSIS (No Button logic - Direct load)
        st.divider()
        st.markdown(f"## ❤️ Interest Analysis for {person_name}")

        # Calling your helper function as it is
        result = helper.calculate_interest(df, person_name)

        st.subheader(f"Interest Level: {result['level']}")
        st.write(f"Score: {result['interest_score']}")

        if 'details' in result:
            st.write("### 📊 Factors:")
            st.json(result['details'])

            # 📈 Interest Over Time
            st.markdown("### 📈 Interest Over Time")
            i_timeline = helper.interest_over_time(df, person_name)
            fig, ax = plt.subplots()
            ax.plot(i_timeline.index.astype(str), i_timeline.iloc[:, 0], color='red')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # 🔍 Extra Insights
            st.markdown("### 🔍 Extra Insights")
            factors = helper.interest_factors(df, person_name)
            st.json(factors)

else:
    st.info("Please upload a WhatsApp chat file (.txt) from the sidebar.")