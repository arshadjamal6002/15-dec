import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set a pleasant theme with Streamlit
st.set_page_config(page_title="Whatsapp Chat Analyzer", page_icon="💖", layout="wide")
st.sidebar.title("add sidebar title here")

# Updated Custom Styling
# Updated Custom Styling
st.sidebar.markdown("""
    <style>
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #D7EFFF; /* Light blue color */
        color: black; /* Black text color */
        border-radius: 10px;
        padding: 10px;
    }
    section[data-testid="stSidebar"] label {
        color: black; /* Ensure labels and titles in the sidebar are black */
    }

    /* Main Area Styling */
    div[data-testid="stAppViewContainer"] {
        background-color: white; /* White background for the main app */
        color: black; /* Black text for the main app */
    }
    div[data-testid="stMetric-value"] {
        color: black !important; /* Ensure metrics' values are black */
    }

    /* Header Styling */
    header[data-testid="stHeader"] {
        background-color: #F0F0F0; /* Light gray header */
        color: black; /* Black text for header */
    }

    /* Button Styling */
    .stButton>button {
        background-color: #FF69B4; /* Keep button pink for contrast */
        color: white; /* White text for buttons */
        border-radius: 10px;
        border: none;
        padding: 5px 15px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)



uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt"])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        
        # Fetch stats for the loudest and quietest users
        loudest_data, quietest_data = helper.get_loudest_and_quietest_users(df)

        # Longest and Shortes Messages
        longest_user, longest_length, shortest_user, shortest_length = helper.message_length_stats(selected_user, df)

        # Media Stats (Pictures, Audio, etc.)
        num_images, num_audio, num_video = helper.media_stats(selected_user, df)
        
        st.title("✨ stats 🎉")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"<h3 style='color:black;'>Total Messages</h3><h4 style='color:black;'>{num_messages}</h4>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h3 style='color:black;'>Total Words</h3><h4 style='color:black;'>{words}</h4>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<h3 style='color:black;'>Media Shared</h3><h4 style='color:black;'>{num_media_messages}</h4>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<h3 style='color:black;'>Links Shared</h3><h4 style='color:black;'>{num_links}</h4>", unsafe_allow_html=True)
       
        # Display Longest and Shortest Message Stats
        st.subheader("Message Length Stats")
        st.write(f"Longer (gyaani) messages: {longest_user} with an average length of {longest_length} words")
        st.write(f"Shorter (cute) messages: {shortest_user} with an average length of {shortest_length} words")

        # # Display Media Stats
        # st.subheader("Media Sharing Stats")
        # st.write(f"Number of Images Shared: {num_images}")
        # st.write(f"Number of Audio Files Shared: {num_audio}")
        # st.write(f"Number of Videos Shared: {num_video}")

        # Generate the reply network graph
        G = helper.reply_network(selected_user, df)
        
        # Display the network plot
        st.title("📶 Reply Network")
        helper.plot_reply_network(G)
        st.markdown("<h4 style='color: #5A8EE3;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        # Generate and display the steamgraph
        st.title("📊 Message Activity Over Time (Loudest and Quietest Users)")
        fig = helper.plot_steamgraph(loudest_data, quietest_data)
        st.plotly_chart(fig)
        st.markdown("<h4 style='color: #5A8EE3;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        # Monthly Timeline
        st.title("📅 Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(timeline['time'], timeline['message'], color='gold', linewidth=2)
        ax.set_xticklabels(timeline['time'], rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("<h4 style='color: #5A8EE3;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        # Daily Timeline
        st.title("📅 Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='violet', linewidth=2)
        ax.set_xticklabels(daily_timeline['only_date'], rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("<h4 style='color: #5A8EE3;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        # Activity Map
        st.title("🗓 Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(busy_day.index, busy_day.values, color='pink')
            ax.set_xticklabels(busy_day.index, rotation=45, ha='right')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(busy_month.index, busy_month.values, color='orchid')
            ax.set_xticklabels(busy_month.index, rotation=45, ha='right')
            st.pyplot(fig)
        st.markdown("<h4 style='color: #FFC0CB;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        # Weekly Activity Heatmap
        st.title("📊 Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.heatmap(user_heatmap, annot=True, fmt="d", cmap="coolwarm", linewidths=0.5)
        st.pyplot(fig)
        st.markdown("<h4 style='color: #5A8EE3;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        # Most Busy Users in the Group (Overall level)
        if selected_user == 'Overall':
            st.title("🔥 Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots(figsize=(10, 6))

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='salmon')
                ax.set_xticklabels(x.index, rotation=45, ha='right')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
                st.markdown("<h4 style='color: black;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        # Wordcloud
        st.title("🌸 Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(df_wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        st.markdown("<h4 style='color: #45902A;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        # Most Common Words
        st.title("🔠 Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(most_common_df[0], most_common_df[1], color='orchid')
        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_ylabel("Words", fontsize=12)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        st.markdown("<h4 style='color: #E96CDA;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        st.title("📈 Word Usage Trends")
        words_of_interest = ['love', 'anger', 'happy', 'sad']
        # Generate trends
        word_trend_df = helper.word_frequency_trend(df, words_of_interest, time_interval="month")
        
        # Choose between Matplotlib or Seaborn
        st.header("Matplotlib Visualization")
        helper.plot_word_trends_matplotlib(word_trend_df, words_of_interest)
        st.markdown("<h4 style='color: #5DCA31;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)



        st.title("📊 Sentiment Comparison Over Time")

        # Aggregating sentiment scores for both methods
        sentiment_hu_liu = helper.aggregate_sentiment_by_user_and_date(df, helper.hu_liu_sentiment)
        sentiment_breen = helper.aggregate_sentiment_by_user_and_date(df, helper.breen_sentiment_score)
        
        # Merge both sentiment dataframes
        sentiment_comparison = pd.merge(sentiment_hu_liu, sentiment_breen, on=['user', 'date'], suffixes=('_hu_liu', '_breen'))

        # Plot the comparison using Matplotlib
        plt.figure(figsize=(12, 6))

        # Plot for each user
        for user in sentiment_comparison['user'].unique():
            user_data = sentiment_comparison[sentiment_comparison['user'] == user]
            
            # Hu & Liu sentiment plot
            plt.scatter(user_data['date'], user_data['sentiment_hu_liu'], label=f"{user} (Hu & Liu)", s=50, marker='o', color='blue')
            
            # Breen sentiment plot
            plt.scatter(user_data['date'], user_data['sentiment_breen'], label=f"{user} (Breen)", s=50, marker='x', color='red')

        plt.title('Sentiment Comparison (Hu & Liu vs Breen)', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Sentiment Score', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)  # Render plot in Streamlit
        st.markdown("<h4 style='color: #EE7676;'>This shows that most of our convos are neautral, hence the 0 sentiment score, but predominatnly we are on the positve side which shows they are mostly happy nd a little sad more in the period from August 23 to january 2024 maybe coz i was leaving nd then it was unclear the situation between us coz of the septum. Also the places where cross and dots are together means we agreed with each others views(which we did most of the times) </h4>", unsafe_allow_html=True)


        # Emoji Analysis
        st.title("😀 Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
            st.markdown("<h4 style='color: black;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(emoji_df['Count'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f", colors=sns.color_palette("Set2", len(emoji_df['Emoji'].head())))
            st.pyplot(fig)

