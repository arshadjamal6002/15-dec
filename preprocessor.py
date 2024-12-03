import re
import pandas as pd

def preprocess(data):
    # Define the regex pattern for the date and time format
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    # Split the data into messages and extract the corresponding dates
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Create a DataFrame from the extracted messages and dates
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert message_date into a datetime object with the correct format
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M - ')

    # Rename 'message_date' to 'date'
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Initialize lists for users and messages
    users = []
    messages = []

    # Process each message and split the user and the message
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]:  # If there's a user in the message
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:  # Handle group notifications or messages without a user
            users.append('group_notification')
            messages.append(entry[0])

    # Add 'user' and 'message' columns to the DataFrame
    df['user'] = users
    df['message'] = messages

    # Drop the 'user_message' column as it is no longer needed
    df.drop(columns=['user_message'], inplace=True)

    # Extract date-related features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Generate period (e.g., 'hour-xx')
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df
