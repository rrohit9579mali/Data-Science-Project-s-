import re
import pandas as pd

def preprocess(data):
    # 🔥 Step 1: Regex Pattern - Isme humne flexible rakha hai
    # Taaki '1/3/25' (2-digit) aur '3/14/2026' (4-digit) dono capture ho jayein.
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # DataFrame creation
    df = pd.DataFrame({
        'user_message': messages,
        'message_date': dates
    })

    # Cleaning date string
    df['message_date'] = df['message_date'].str.replace(' -', '', regex=False).str.strip()

    # 🔥 Step 2: Super Smart Date Conversion
    # 'dayfirst=False' rakha hai kyunki 3/14 format US style (MM/DD/YY) lag raha hai.
    # Ye messages ko miss hone se bachaega.
    df['date'] = pd.to_datetime(df['message_date'], dayfirst=False, errors='coerce')

    # Backup attempt: Agar upar wala fail ho toh default format try karega
    mask = df['date'].isna()
    df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'message_date'], errors='coerce')

    # Remove only those rows where date is absolutely unreadable
    df = df.dropna(subset=['date'])

    # 🔥 Step 3: CRITICAL FILTER (No Future Data)
    # Isse 2027 wala bug solve ho jayega
    df = df[df['date'].dt.year <= 2026]

    # Step 4: Separate User & Message
    users = []
    messages = []

    for message in df['user_message']:
        # maxsplit=1 use kiya hai taaki message ke andar ke ":" se split na ho
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) > 1:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message', 'message_date'], inplace=True)

    # 🔥 Step 5: Feature Engineering
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Time period generation
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append(f"00-{hour+1}")
        else:
            period.append(f"{hour}-{hour+1}")

    df['period'] = period

    return df