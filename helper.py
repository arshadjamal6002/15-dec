from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import numpy as np
import streamlit as st

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head(2)
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user, df):
    """
    Generate a circular word cloud for a selected user from the chat data.
    
    Parameters:
    selected_user (str): The user to filter messages for. Use 'Overall' for all users.
    df (pd.DataFrame): The chat data containing 'user' and 'message' columns.

    Returns:
    None: Displays the word cloud.
    """
    # Read stop words
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().splitlines()

    # Filter messages for the selected user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Remove group notifications and media messages
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    # Remove stop words from messages
    def remove_stop_words(message):
        y = [word for word in message.lower().split() if word not in stop_words]
        return " ".join(y)

    temp['message'] = temp['message'].apply(remove_stop_words)

    # Load circular mask
    x, y = np.ogrid[:500, :500]
    mask = (x - 250) ** 2 + (y - 250) ** 2 > 250 ** 2
    mask = 255 * mask.astype(int)

    # Generate word cloud
    wc = WordCloud(width=500, height=500, mask=mask, 
                   contour_width=2, contour_color='white', 
                   background_color='white', min_font_size=10)

    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        # Extract emojis using emoji.emoji_list method
        emoji_list = emoji.emoji_list(message)
        emojis.extend([e['emoji'] for e in emoji_list])  # Collect emojis from the list

    # Count the occurrences of each emoji
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    emoji_df.columns = ['Emoji', 'Count']  # Rename columns for clarity

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap




# add ons
def message_length_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Calculate the length of each message (you can use len(message) for character count)
    df['message_length'] = df['message'].apply(lambda x: len(x.split()))  # word count

    # Calculate average message length for each user
    user_message_lengths = df.groupby('user')['message_length'].mean()

    # Get the user with the longest and shortest messages
    longest_user = user_message_lengths.idxmax()
    shortest_user = user_message_lengths.idxmin()

    return longest_user, user_message_lengths[longest_user], shortest_user, user_message_lengths[shortest_user]


def media_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Count number of media messages per user
    media_messages = df[df['message'] == '<Media omitted>\n']
    
    # Count different types of media
    num_images = media_messages[media_messages['message'].str.contains("image", case=False)].shape[0]
    num_audio = media_messages[media_messages['message'].str.contains("audio", case=False)].shape[0]
    num_video = media_messages[media_messages['message'].str.contains("video", case=False)].shape[0]

    return num_images, num_audio, num_video



import networkx as nx
import matplotlib.pyplot as plt

def reply_network(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['user'] != 'group_notification']   
    # Create an empty directed graph
    G = nx.DiGraph()

    # Iterate through the messages and track replies
    for i in range(1, df.shape[0]):
        current_user = df.iloc[i]['user']
        prev_user = df.iloc[i - 1]['user']
        
        # If the current message is a reply to the previous message, add an edge
        if current_user != prev_user:
            if G.has_edge(prev_user, current_user):
                G[prev_user][current_user]['weight'] += 1
            else:
                G.add_edge(prev_user, current_user, weight=1)

    return G


def plot_reply_network(G):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15, iterations=20)  # Layout for the graph
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_color="black")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Display the graph using Streamlit
    st.pyplot(plt)


def get_loudest_and_quietest_users(df):
    # Filter out group notifications and media messages
    df = df[df['user'] != 'group_notification']
    df = df[df['message'] != '<Media omitted>\n']

    # Group by month and user and get the count of messages per user per month
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    user_monthly_activity = df.groupby(['user', 'year_month']).size().reset_index(name='message_count')

    # Find the top 2 loudest users (most messages)
    loudest_users = user_monthly_activity.groupby('user')['message_count'].sum().nlargest(2).index.tolist()

    # Find the 2 quietest users (least messages)
    quietest_users = user_monthly_activity.groupby('user')['message_count'].sum().nsmallest(2).index.tolist()

    # Filter data for the loudest and quietest users
    loudest_data = user_monthly_activity[user_monthly_activity['user'].isin(loudest_users)]
    quietest_data = user_monthly_activity[user_monthly_activity['user'].isin(quietest_users)]

    return loudest_data, quietest_data

import plotly.express as px

def plot_steamgraph(loudest_data, quietest_data):
    # Combine loudest and quietest user data
    data = pd.concat([loudest_data, quietest_data])

    # Create a steamgraph-like plot (area chart)
    fig = px.area(data, x="year_month", y="message_count", color="user", line_group="user", 
                  title="Message Activity of Loudest and Quietest Users Over Time")

    fig.update_layout(xaxis_title="Time", yaxis_title="Message Count", template="plotly_dark")
    return fig


# Example function to generate word trends
def word_frequency_trend(df, words, time_interval="month"):
    """
    Generate trends for word frequencies over a time interval.

    Parameters:
    df (pd.DataFrame): The chat data.
    words (list): Words to track.
    time_interval (str): The time grouping interval ("month" or "day").

    Returns:
    pd.DataFrame: A DataFrame with time and word frequencies.
    """
    trend_data = []
    grouped_df = df.groupby([time_interval])['message'].apply(list)

    for time, messages in grouped_df.items():
        word_counts = {word: 0 for word in words}
        for message in messages:
            for word in words:
                word_counts[word] += message.lower().split().count(word)
        word_counts["time"] = time
        trend_data.append(word_counts)

    return pd.DataFrame(trend_data)

# Matplotlib plot
def plot_word_trends_matplotlib(word_trend_df, words):
    plt.figure(figsize=(12, 6))
    for word in words:
        plt.plot(word_trend_df['time'], word_trend_df[word], label=word, marker='o')
    plt.title("Word Usage Trends Over Time", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title="Words", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)  # Streamlit display

# Example sentiment lexicons
hu_liu_positive_words = ['good', 'happy', 'joy', 'love']  # Add your lexicon words here
hu_liu_negative_words = ['bad', 'angry', 'hate', 'sad']  # Add your lexicon words here

# Sentiment scoring function from Jeffrey Breen (Example)
def breen_sentiment_score(message):
    """
    Example sentiment scoring function based on Jeffrey Breen's method.
    A simple positive/negative word-based score.
    """
    positive_words = ['good', 'happy', 'joy', 'love']  # Add your lexicon
    negative_words = ['bad', 'angry', 'hate', 'sad']  # Add your lexicon
    
    positive_score = sum([message.lower().count(word) for word in positive_words])
    negative_score = sum([message.lower().count(word) for word in negative_words])
    
    return positive_score - negative_score  # Basic difference, can be expanded

# Function to calculate sentiment using Hu & Liu lexicon
def hu_liu_sentiment(message):
    """
    Calculate sentiment using Hu and Liu lexicon (simple positive/negative word counts).
    """
    positive_score = sum([message.lower().split().count(word) for word in hu_liu_positive_words])
    negative_score = sum([message.lower().split().count(word) for word in hu_liu_negative_words])
    return positive_score - negative_score

# Function to aggregate sentiment scores by user and date
def aggregate_sentiment_by_user_and_date(df, sentiment_func):
    """
    Aggregate sentiment scores by user and date.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing messages and dates.
    sentiment_func (function): The sentiment function to use (Hu & Liu or Breen).
    
    Returns:
    pd.DataFrame: Dataframe with aggregated sentiment scores by user and date.
    """
    df['sentiment'] = df['message'].apply(sentiment_func)
    df['date'] = pd.to_datetime(df['only_date']).dt.date  # Extract date only
    sentiment_by_user_and_date = df.groupby(['user', 'date'])['sentiment'].mean().reset_index()
    return sentiment_by_user_and_date

