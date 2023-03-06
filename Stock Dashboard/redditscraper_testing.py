import praw
import scraper_keys
import pandas as pd

user_agent = "Scraper Version Test 1.0"
reddit = praw.Reddit(
    client_id=scraper_keys.client_id,
    client_secret=scraper_keys.client_secret,
    user_agent=user_agent,
)

ticker_count = dict()
for submission in reddit.subreddit("wallstreetbets").search(
    query="$", time_filter="month", sort="hot"
):
    # valid_checker = list(submission.title)
    # if "$" in valid_checker:  # check to see if the title has a $ in it
    submission_list = submission.title.split()  # isolate the ticker in a list
    for ticker in submission_list:
        if (
            ticker.startswith("$")
            and not any(char.isdigit() for char in ticker)
            and len(ticker) > 1
        ):  # find ticker, and make sure it's valid
            if not ticker in ticker_count:
                ticker_count[ticker] = 1  # add to dictionary for counting
            else:
                ticker_count[ticker] += 1
    df = pd.DataFrame.from_dict(ticker_count, orient="index")

print(df)
