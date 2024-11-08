from flask import Flask, jsonify, request, Response, send_file
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
import io
import zipfile
import requests
import os
import traceback
from urllib.parse import urlparse, unquote

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

@app.route('/download_images', methods=['POST'])
def download_images():
    try:
        data = request.get_json()
        image_urls = data.get('image_urls', [])
        if not image_urls:
            return jsonify({'error': 'No image URLs provided.'}), 400

        zip_buffer = io.BytesIO()
        successful_downloads = False

        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for index, url in enumerate(image_urls):
                try:
                    if not re.match(r'^https?://', url):
                        continue

                    response = requests.get(url, stream=True, timeout=10)
                    response.raise_for_status()

                    ext = url.split('.')[-1]
                    if ext.lower() not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                        ext = 'jpg'

                    image_name = f'image_{index + 1}.{ext}'
                    zip_file.writestr(image_name, response.content)
                    successful_downloads = True

                except requests.exceptions.RequestException as e:
                    print(f'Error downloading {url}: {e}')
                    continue

        if not successful_downloads:
            return jsonify({'error': 'No valid images found.'}), 404

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='images.zip'
        )
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({'error': 'Internal server error.'}), 500
    
@app.route('/download_content', methods=['POST'])
def download_content():
    try:
        data = request.get_json()
        posts = data.get('posts', [])
        subreddit = data.get('subreddit', 'default_subreddit')
        if not posts:
            return jsonify({'error': 'No posts provided.'}), 400

        zip_buffer = io.BytesIO()
        successful_downloads = False

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for post_index, post in enumerate(posts):
                try:
                    title = post.get('title') or 'Untitled Post'
                    pub_date = post.get('date') or 'Unknown Date'
                    text_content = post.get('text_content') or ''
                    image_urls = post.get('image_urls', [])

                    print(f"Processing post {post_index}: {title}")

                    post_has_content = False

                    if isinstance(text_content, str) and text_content.strip():
                        post_has_content = True

                    images_downloaded = False
                    images_data = []

                    for index, url in enumerate(image_urls):
                        try:
                            if not re.match(r'^https?://', url):
                                continue

                            response = requests.get(url, stream=True, timeout=10)
                            response.raise_for_status()

                            parsed_url = urlparse(url)
                            path = parsed_url.path
                            filename = os.path.basename(path)
                            name, ext = os.path.splitext(filename)
                            ext = ext.lstrip('.').lower()

                            if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                                ext = 'jpg'

                            image_name = f"image_{index + 1}.{ext}"
                            images_data.append({
                                'image_name': image_name,
                                'content': response.content
                            })
                            images_downloaded = True

                        except requests.exceptions.RequestException as e:
                            print(f'Error downloading {url}: {e}')
                            continue

                    print(f"Downloaded {len(images_data)} images for post {post_index}")

                    if post_has_content or images_downloaded:
                        folder_name = f"{title}_{pub_date}"
                        folder_name = re.sub(r'[\\/*?:"<>|]', "_", folder_name)

                        if post_has_content:
                            text_file_path = os.path.join(folder_name, 'content.txt')
                            zip_file.writestr(text_file_path, text_content)

                        for image_data in images_data:
                            image_path = os.path.join(folder_name, image_data['image_name'])
                            zip_file.writestr(image_path, image_data['content'])

                        successful_downloads = True

                except Exception as e:
                    print(f"Error processing post at index {post_index}: {e}")
                    traceback.print_exc()
                    continue

        if not successful_downloads:
            return jsonify({'error': 'No valid content found.'}), 404

        subreddit = re.sub(r'[\\/*?:"<>|]', "_", subreddit)

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{subreddit}_posts.zip'
        )
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    app.run(port=5000, threaded=True)
