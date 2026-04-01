from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob

extract = URLExtract()

# 📊 Stats
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media, len(links)


# 👥 Busy Users
def most_busy_users(df):
    x = df['user'].value_counts().head()
    percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    return x, percent


# ☁️ WordCloud
def create_wordcloud(df):
    wc = WordCloud(width=600, height=400, background_color='white')
    return wc.generate(df['message'].str.cat(sep=" "))


#  Emoji
def emoji_helper(df):
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    return pd.DataFrame(Counter(emojis).most_common(20), columns=['emoji', 'count'])


#  Common Words
def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    words = []
    for message in df['message']:
        words.extend(message.lower().split())

    stop_words = set(["the", "is", "and", "to", "a", "of", "in"])
    words = [w for w in words if w not in stop_words]

    return pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])


# Timeline
def monthly_timeline(df):
    df['month'] = df['date'].dt.month_name()
    df['year'] = df['date'].dt.year
    df['time'] = df['month'] + "-" + df['year'].astype(str)
    return df.groupby('time').count()['message'].reset_index()


def daily_timeline(df):
    return df.groupby(df['date'].dt.date).count()['message'].reset_index()


#  Activity
def week_activity_map(df):
    return df['date'].dt.day_name().value_counts()


def month_activity_map(df):
    return df['date'].dt.month_name().value_counts()


def activity_heatmap(df):
    return df.pivot_table(index=df['date'].dt.day_name(),
                          columns=df['date'].dt.hour,
                          values='message',
                          aggfunc='count').fillna(0)


#  Sentiment
def sentiment_analysis(df):
    sentiments = []
    for msg in df['message']:
        try:
            score = TextBlob(msg).sentiment.polarity
            if score > 0:
                sentiments.append('Positive')
            elif score < 0:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')
        except:
            sentiments.append('Neutral')
    df['sentiment'] = sentiments
    return df


def sentiment_counts(df):
    return df['sentiment'].value_counts()


# ⏱️ Hourly
def hourly_activity(df):
    return df['date'].dt.hour.value_counts().sort_index()


# 🔗 Links
def top_links(df):
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    return Counter(links).most_common(10)