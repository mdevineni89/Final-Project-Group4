import pandas as pd
from pmaw import PushshiftAPI
import os

api = PushshiftAPI(num_workers=16)


comments = api.search_comments(
    subreddit='cryptocurrency',
    fields=['created_utc', 'author', 'body'],
    after=int(pd.to_datetime('5/1/2021').timestamp()),
    before=int(pd.to_datetime('4/1/2022').timestamp())
)
comments_df = pd.DataFrame(comments)
comments_df = comments_df.sort_values(by=['created_utc'])
comments_df.to_csv('cryptocurrency_comments_raw.csv', index=False)

