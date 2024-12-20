import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set a pleasant theme with Streamlit
st.set_page_config(page_title="Whatsapp Chat Analyzer", page_icon="💖", layout="wide")
st.sidebar.title("Happy Birthday Ladi Blossom 💖")

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
        
        st.title("✨ Birthday stats 🎉")
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
        st.markdown("<h4 style='color: #BDDFEB;'>The graph represents a highly active one-on-one conversation between us, nd the size of the nodes represents the contributions of the users which are relatively equal in our case...</h4>", unsafe_allow_html=True)


        # Generate and display the steamgraph
        st.title("📊 Message Activity Over Time (Loudest and Quietest Users)")
        fig = helper.plot_steamgraph(loudest_data, quietest_data)
        st.plotly_chart(fig)
        st.markdown("""
        <ul style="color: #5A8EE3;">
            <li>Both of us exhibit periods of high activity, with visible spikes in certain months.</li>
            <li>Activity is largely aligned, suggesting that our message counts rise and fall together, which indicates responsive conversations or shared periods of engagement. Aww!</li>
            <li>The highest peaks occurred around early 2024, where My dearest PEACHIE-PIE has slightly more messages than me myself, very shockingly indicating my bubs contributes slightly more at peak times. Yaar tbh, I never knew this yaar... Thx, bubs!</li>
            <li>Consistent interaction with matching trends highlights a strong conversational relationship, with neither of us significantly dominating overall.. hehe</li>
        </ul>
        """, unsafe_allow_html=True)


        # Monthly Timeline
        st.title("📅 Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(timeline['time'], timeline['message'], color='gold', linewidth=2)
        ax.set_xticklabels(timeline['time'], rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("""
        <ul style="color: #FFD808;">
            <li><b>Relationship Growth:</b> There is a gradual and consistent increase in activity over time, with a significant surge starting in early 2024.</li>
            <li><b>Sharp Peak:</b> The chart exhibits a dramatic spike in message activity, reaching its highest point in mid-2024 (above 17,500 messages).</li>
            <li><b>Stability and Drop:</b> Following the peak, the activity stabilizes at a slightly lower level before a sharp decline toward the end of 2024 maybe coz we started being a less lovey dovey nd focussing more on our interships and padhai and also restricting chats to calls at the time i was out of my home.</li>
            <li><b>Quiet Start:</b> The early periods (2022–2023) show relatively low and consistent message counts, indicating minimal activity.</li>
            <li><b>Breakthrough Year:</b> 2024 marks a breakthrough in our activity, with rapid and significant growth compared to previous years. thanks bub for comin into ma world!</li>
        </ul>
        """, unsafe_allow_html=True)


        # Daily Timeline
        st.title("📅 Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='violet', linewidth=2)
        ax.set_xticklabels(daily_timeline['only_date'], rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("""
        - **Gradual Start:** The timeline begins with low and consistent message activity from June 2022 to July 2022 since we were just in the phases of close friends.
        - **Initial Growth:** A noticeable increase starts around late July 2022, with activity rising steadily as this time u started to tel me abt your septum thingy
        - **Consistent Spikes:** From August 2022 to early January 2023, there are frequent spikes, suggesting intermittent bursts of high engagement uk why!
        - **Major Surge:** A dramatic rise in activity occurs around mid-January 2023, where message counts reach new highs, peaking at over 1,000 messages per day. daymmm
        """)


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

        st.markdown("""
        - **Top Days:**  
            - Thursday, Tuesday, and Saturday emerge as the busiest days, showing nearly equal activity levels.  
            - Friday, Monday, and Sunday follow closely with slightly lower message counts cz usual you dont wanna get up on weekends....zzz  
        - **Lowest Activity Day:**  
            - Wednesday has the least activity compared to other days.(hmm thats interesting to think abt)
        """)

        st.markdown("""
        - **Peak Month:**  
            - **July** records the highest activity, with over 17,000 messages, making it the most engaged month. coz it all started in july nd i also came to see ya in july after my exchange  
        - **Other Busy Months:**  
            - November, October, and September also display high activity, with significant message counts.  
        - **Gradual Decline:**  
            - Activity drops steadily from August to May, with **May** showing the lowest engagement cz in May we were just beginning to have convos.. ywt im glad i did turned out to be the most beautiful ladi with the most beautiful soul.
        """)


        # Weekly Activity Heatmap
        st.title("📊 Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.heatmap(user_heatmap, annot=True, fmt="d", cmap="coolwarm", linewidths=0.5)
        st.pyplot(fig)
        st.markdown("""
        - **Peak Activity Hours:**  
            - Activity is highest between **10 PM and 12 AM**, particularly on Fridays and Sundays, as indicated by the red zones in the heatmap.
        
        - **Consistent Evening Engagement:**  
            - A noticeable level of engagement occurs between **8 PM and 10 PM** across all days, MHMMM

        - **Low Activity Periods:**  
            - Early morning hours, specifically **4 AM to 7 AM**, exhibit minimal activity across all days coz someone doesnt wanna wake up early!
            - Activity gradually increases after **10 AM**, with consistent engagement throughout the afternoon and evening.

        - **Day-Specific Trends:**  
            - **Friday** and **Sunday** have the most notable spikes in activity late at night.
            - **Wednesday** and **Tuesday** display comparatively lower activity overall.

        - **Overall Pattern:**  
            - Weekday activity is relatively consistent, while weekends see noticeable peaks late at night. coz dinbhar she sleeps...
        """)

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
                st.markdown("<h4 style='color: black;'>UK what, i used to think that im the one who is most active on chats but oh my ladi blossom even proved me wrong in this as well.</h4>", unsafe_allow_html=True)


        # Wordcloud
        st.title("🌸 Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(df_wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        st.markdown("<h4 style='color: #1fa187;'>The most vivid inference u can get from this is out of all words weve exchanged it all STILL REVOLVES AROUND *U*!</h4>", unsafe_allow_html=True)


        # Most Common Words
        st.title("🔠 Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(most_common_df[0], most_common_df[1], color='orchid')
        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_ylabel("Words", fontsize=12)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        st.markdown("<h4 style='color: #da70d6;'>*YAAR* is the most common of all words amongst us resembling that amidst this love we still are forever friends forever the study partners we were in the beginning.</h4>", unsafe_allow_html=True)



        st.title("📈 Word Usage Trends")
        words_of_interest = ['love', 'anger', 'happy', 'sad']
        # Generate trends
        word_trend_df = helper.word_frequency_trend(df, words_of_interest, time_interval="month")
        
        # Choose between Matplotlib or Seaborn
        st.header("Matplotlib Visualization")
        helper.plot_word_trends_matplotlib(word_trend_df, words_of_interest)
        st.markdown("<h4 style='color: #2ca02c;'>Out of everything else , Love continued to grow all thru out nd wd always bubs</h4>", unsafe_allow_html=True)




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
        st.markdown("<h4 style='color: #ff0000;'>This shows that most of our convos are neautral, hence the 0 sentiment score, but predominatnly we are on the positve side which shows they are mostly happy nd a little sad more in the period from August 23 to january 2024 maybe coz i was leaving nd then it was unclear the situation between us coz of the septum. Also the places where cross and dots are together means we agreed with each others views(which we did most of the times) </h4>", unsafe_allow_html=True)


        # Emoji Analysis
        st.title("😀 Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
            st.markdown("<h4 style='color: black;'>the most common used emoji is still the one which ive been using forever...</h4>", unsafe_allow_html=True)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(emoji_df['Count'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f", colors=sns.color_palette("Set2", len(emoji_df['Emoji'].head())))
            st.pyplot(fig)

