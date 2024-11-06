from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import feedparser
from textblob import TextBlob
from bs4 import BeautifulSoup
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import text2emotion as te
import threading
import time

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)

progress_info = {}
results_info = {}
lock = threading.Lock()

def process_feed(subreddit, limit):
    try:
        feed_url = f'https://www.reddit.com/r/{subreddit}.rss?limit={limit}'
        feed = feedparser.parse(feed_url)

        with lock:
            progress_info[subreddit]['total_posts'] = len(feed.entries)

        posts = []

        for i, entry in enumerate(feed.entries, 1):
            with lock:
                progress_info[subreddit]['current_post'] = i

            title_sentiment = TextBlob(entry.title).sentiment.polarity

            summary_text = ""
            if 'summary' in entry:
                soup = BeautifulSoup(entry.summary, 'html.parser')
                md_div = soup.find('div', class_='md')
                summary_text = md_div.get_text() if md_div else ''

            summary_sentiment = TextBlob(summary_text).sentiment.polarity

            emotions = te.get_emotion(summary_text)

            sentences = sent_tokenize(summary_text)
            num_sentences = len(sentences) if len(sentences) > 0 else 1

            words = word_tokenize(summary_text)
            num_words = len(words)

            unique_words = set(words)
            lexical_diversity = len(unique_words) / num_words if num_words > 0 else 0

            total_characters = sum(len(word) for word in words)
            avg_word_length = total_characters / num_words if num_words > 0 else 0

            avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

            filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalnum()]
            word_counts = Counter(filtered_words)
            top_words = [word for word, count in word_counts.most_common(13)]

            post = {
                'subreddit': subreddit,
                'title': entry.title,
                'link': entry.link,
                'summary': entry.summary if 'summary' in entry else '',
                'pubDate': entry.published if 'published' in entry else 'No publish date',
                'titleSentiment': 'positive' if title_sentiment > 0 else 'negative' if title_sentiment < 0 else 'neutral',
                'summarySentiment': 'positive' if summary_sentiment > 0 else 'negative' if summary_sentiment < 0 else 'neutral',
                'keywords': top_words,
                'emotions': emotions,
                'totalWordCount': num_words,
                'lexicalDiversity': lexical_diversity,
                'avgWordLength': avg_word_length,
                'avgSentenceLength': avg_sentence_length
            }
            posts.append(post)

        with lock:
            results_info[subreddit] = posts
            progress_info.pop(subreddit, None)
    except Exception as e:
        with lock:
            progress_info.pop(subreddit, None)
        print(f"Error processing subreddit '{subreddit}': {e}")

@app.route('/fetch_reddit_feed', methods=['GET'])
def fetch_reddit_feed():
    subreddit = request.args.get('subreddit')
    limit = request.args.get('limit', '100')

    if not subreddit:
        return jsonify({"error": "Subreddit name is required"}), 400

    with lock:
        progress_info[subreddit] = {'current_post': 0, 'total_posts': 0}

    thread = threading.Thread(target=process_feed, args=(subreddit, limit))
    thread.start()

    return jsonify({"message": "Processing started"}), 202

@app.route('/progress_stream', methods=['GET'])
def progress_stream():
    subreddit = request.args.get('subreddit')
    if not subreddit:
        return jsonify({"error": "Subreddit name is required"}), 400

    def generate():
        while True:
            with lock:
                if subreddit in progress_info:
                    progress = progress_info[subreddit]
                    yield f"data:{progress['current_post']} of {min(progress['total_posts'], 100)} posts processed\n\n"
                elif subreddit in results_info:
                    yield "data:complete\n\n"
                    break
                else:
                    yield "data:unknown\n\n"
                    break
            time.sleep(1)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/get_results', methods=['GET'])
def get_results():
    subreddit = request.args.get('subreddit')
    if not subreddit:
        return jsonify({"error": "Subreddit name is required"}), 400

    with lock:
        if subreddit in results_info:
            return jsonify(results_info.pop(subreddit))
        else:
            return jsonify({"message": "Results not ready"}), 202

if __name__ == '__main__':
    app.run(port=5000, threaded=True)
