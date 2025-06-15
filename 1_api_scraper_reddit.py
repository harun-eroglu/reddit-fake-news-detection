import praw
import pandas as pd

# Reddit API
client_id = "---"
client_secret = "---"
user_agent = "---"

reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Subreddit configuration: (subreddit_name, label)
subreddits = [
    ('news', 0),
    ('worldnews', 0),
    ('science', 0),
    ('environment', 0),
    ('technology', 0),
    ('conspiracy', 1),
    ('UFOs', 1),
    ('conspiracytheories', 1),
    ('Paranormal', 1),
    ('aliens', 1)
]

# Fetch posts by category (new, hot, top)
def fetch_posts_by_category(subreddit_name, label, category, limit=400):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    if category == 'new':
        submissions = subreddit.new(limit=limit)
    elif category == 'hot':
        submissions = subreddit.hot(limit=limit)
    elif category == 'top':
        submissions = subreddit.top(limit=limit)
    else:
        raise ValueError("Category must be 'new', 'hot' or 'top'")
    
    for submission in submissions:
        if not submission.stickied and (submission.selftext or submission.title):
            text = (submission.title or '') + ' ' + (submission.selftext or '')
            posts.append({
                'id': submission.id,
                'text': text.strip(),
                'subreddit': subreddit_name,
                'label': label,
                'source': category
            })
    return pd.DataFrame(posts)

# Collect all data
df_list = []
for name, label in subreddits:
    print(f"Fetching from r/{name}...")
    try:
        dfs = []
        for category in ['new', 'hot', 'top']:
            df_cat = fetch_posts_by_category(name, label, category, limit=400)
            print(f"  Collected {len(df_cat)} posts from r/{name} ({category})")
            dfs.append(df_cat)
        df_sub = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='id')
        print(f"Total unique posts for r/{name}: {len(df_sub)}")
        df_list.append(df_sub)
    except Exception as e:
        print(f"Failed to fetch from r/{name}: {e}")

df_all = pd.concat(df_list, ignore_index=True)
df_all = df_all.drop_duplicates(subset='id')
df_all = df_all.drop(columns=['id'])

df_all.to_csv("1_reddit_dataset_uncleaned.csv", index=False)
print(f"Dataset saved as '1_reddit_dataset_uncleaned.csv' with {len(df_all)} entries.")
