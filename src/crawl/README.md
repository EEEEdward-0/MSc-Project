
# Crawl Module

This module handles Reddit user data crawling. It includes the following scripts:

- `io.py` : Input/output utilities, e.g., saving and loading crawled user data.
- `reddit_client.py` : Wrapper for Reddit API or HTTP requests.
- `runner.py` : Main entry point for running the crawler. Orchestrates the whole process.
- `throttle.py` : Rate limiter to control request speed and avoid hitting Reddit restrictions.
- `user_fetch.py` : Core user fetching logic, including posts and comments.
- `__init__.py` : Package initializer (usually empty).

## Usage
The typical entry point is `runner.py`, which calls the client, fetch, and IO components internally.
