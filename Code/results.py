import pandas as pd
import numpy as np
comments = pd.read_csv('cryptocurrency_comments_cleaned_subset_emotion.csv')
comments['created_datetime'] = pd.to_datetime(comments['created_utc'], unit='s')
comments['date'] = comments['created_datetime'].dt.date


emotion = comments.drop(['created_utc', 'body', 'author', 'created_datetime'], axis=1)
emotion_agg = emotion.groupby(['date', 'emotion']).size().to_frame('num_comments').reset_index()
emotion_agg['percentage_comments'] = 100*emotion_agg['num_comments'] / emotion_agg.groupby(['date'])['num_comments'].transform('sum')
emotion_agg['date'] = pd.to_datetime(emotion_agg['date'])


emotion_pivot = emotion_agg.pivot(index='date', columns='emotion', values='percentage_comments').reset_index()
emotion_pivot = emotion_pivot.fillna(0)


prices = pd.read_csv('cryptocurrency_prices.csv')

prices_emotion = pd.concat([emotion_pivot, prices[['btc']]], axis=1)
# make plots

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='darkgrid')

corr = prices_emotion[['btc','sadness', 'joy', 'love', 'anger', 'fear', 'surprise']].corr()
sns.heatmap(corr, annot=True)
plt.show()

def draw_lineplot(emotion, clr):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('date')
    ax1.set_ylabel('BTC ($)')
    ax1.plot(prices_emotion['date'], prices_emotion['btc'].rolling(7).mean(), color=color)

    ax2 = ax1.twinx()

    color = f'tab:{clr}'
    ax2.set_ylabel(f'Comments with {emotion} (%)', color=color)  # we already handled the x-label with ax1
    ax2.plot(prices_emotion['date'], prices_emotion[emotion].rolling(28).mean(), color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

draw_lineplot('sadness', 'green')
draw_lineplot('joy', 'orange')
draw_lineplot('love', 'pink')
draw_lineplot('anger', 'red')
draw_lineplot('fear', 'gray')
draw_lineplot('surprise', 'purple')

# do corr test
from scipy.stats import pearsonr
pearsonr(emotion_pivot['sadness'], prices['btc'])