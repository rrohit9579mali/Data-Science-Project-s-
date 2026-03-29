from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import os

extract = URLExtract()

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
stop_words_path = os.path.join(current_dir, 'stop_hinglish.txt')

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df_percent

def create_wordcloud(selected_user, df):
    # FIXED: Indentation corrected
    with open(stop_words_path, 'r') as f:
        stop_words = f.read()
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    # FIXED: Path updated here as well
    with open(stop_words_path, 'r') as f:
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

# ... (Baki saare functions: emoji_helper, timelines, maps, activity_heatmap waise hi rakhein)

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    ordered_months = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
    return df['month'].value_counts().reindex(ordered_months).fillna(0)

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def interest_over_time(df, person_name):
    df['temp_month'] = df['date'].dt.to_period('M')
    timeline = df.groupby(['temp_month', 'user']).size().unstack().fillna(0)
    if person_name in timeline.columns:
        timeline['interest_ratio'] = timeline[person_name] / timeline.sum(axis=1)
    else:
        timeline['interest_ratio'] = 0
    return timeline[['interest_ratio']]

def interest_factors(df, person_name):
    person_df = df[df['user'] == person_name]
    avg_msg_len = person_df['message'].apply(lambda x: len(x.split())).mean()
    media_count = person_df[person_df['message'] == '<Media omitted>\n'].shape[0]
    peak_hour = person_df['hour'].mode()[0] if not person_df.empty else 0
    return {
        "Avg Message Length": round(avg_msg_len, 2) if not pd.isna(avg_msg_len) else 0,
        "Media Shared": media_count,
        "Most Active Hour": f"{peak_hour}:00",
        "Unique Days Chatting": person_df['only_date'].nunique()
    }

def calculate_interest(df, person_name):
    df = df.sort_values('date')
    users = df['user'].unique()
    if len(users) < 2:
        return {"level": "Not enough data", "interest_score": 0, "details": {}}
    other = person_name
    me = [u for u in users if u != other][0]
    msg_count = df['user'].value_counts()
    msg_ratio = msg_count.get(other, 0) / msg_count.sum()
    df['reply_time'] = df['date'].diff().dt.total_seconds()
    other_reply = df[df['user'] == other]['reply_time'].mean()
    me_reply = df[df['user'] == me]['reply_time'].mean()
    reply_score = 1 if other_reply < me_reply else 0.5
    df['new_conv'] = df['date'].diff().dt.seconds > 3600
    initiations = df[df['new_conv']]
    init_count = initiations['user'].value_counts()
    init_score = init_count.get(other, 0) / max(1, init_count.sum())
    df['msg_len'] = df['message'].apply(lambda x: len(x.split()))
    avg_len_other = df[df['user'] == other]['msg_len'].mean()
    avg_len_me = df[df['user'] == me]['msg_len'].mean()
    length_score = 1 if avg_len_other >= avg_len_me else 0.5
    df['only_date'] = df['date'].dt.date
    active_days = df.groupby('user')['only_date'].nunique()
    consistency_score = active_days.get(other, 0) / max(1, active_days.sum())
    interest_score = (msg_ratio + reply_score + init_score + length_score + consistency_score) / 5
    if interest_score > 0.5:
        level = "High Interest 💚"
    elif interest_score > 0.3:
        level = "Medium Interest 💛"
    else:
        level = "Low Interest ❤️"
    return {
        "interest_score": round(interest_score, 2),
        "level": level,
        "details": {
            "message_ratio": round(msg_ratio, 2),
            "reply_score": reply_score,
            "initiation_score": round(init_score, 2),
            "length_score": length_score,
            "consistency_score": round(consistency_score, 2)
        }
    }
