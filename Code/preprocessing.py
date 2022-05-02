import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

comments_raw = pd.read_csv('cryptocurrency_comments_raw.csv', engine='python')

# remove bot comments
bots = ['AutoModerator', 'ccModBot', 'coinfeeds-bot']
is_bot = comments_raw['author'].isin(bots)

# remove common comments
common_comments = list(comments_raw[~is_bot]['body'].value_counts()[:100].index)
is_common_comment = comments_raw['body'].isin(common_comments)

# remove nans comments
is_missing = comments_raw['body'].isna()

# remove nans comments
is_duplicated = comments_raw['body'].duplicated()

# save
comments_cleaned = comments_raw[(~is_bot) & (~is_common_comment) & (~is_duplicated) & (~is_missing)]
comments_cleaned_subset = comments_cleaned.sample(frac=0.04)
comments_cleaned_subset.to_csv('cryptocurrency_comments_cleaned_subset.csv', index=False)

# plot
comments_cleaned['date'] = pd.to_datetime(comments_cleaned['created_utc'], unit='s').dt.date
comments_per_day = comments_cleaned.groupby(['date']).size().to_frame('num_comments').reset_index()

comments_per_day['date'] = pd.to_datetime(comments_per_day['date'])
import seaborn as sns
sns.set(style='darkgrid')
sns.lineplot(data=comments_per_day, x='date', y='num_comments')
plt.tight_layout()
plt.show()

