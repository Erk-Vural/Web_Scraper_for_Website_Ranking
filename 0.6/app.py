import random
import sys
from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
from sklearn.metrics import euclidean_distances
import numpy as np

from Crawler import Crawler


def cluster_docs(k=5, word_count=None):
    # transpose tf matrix
    z = []
    for x, y in word_count:
        z.append(y)
    X = np.matrix(z)
    # normalize term frequencies
    X_max, X_min = X.max(), X.min()
    X = (X - X_min) / (X_max - X_min)

    if len(X) < k:
        print("Warning: not enough documents to pick " + str(k) + " leaders.")
        k = int(len(X) / 2)
        print("Clustering around " + str(k) + " leaders.")

    # pick a random sample of k docs to be leaders
    leader_indices = random.sample(range(0, len(X)), k)
    follower_indices = list(set([i for i in range(len(X))]) - set(leader_indices))

    # stores leader: [(follower, distance)]
    clusters = {l: [] for l in leader_indices}

    # assign each follower to its closest leader
    for f in follower_indices:
        min_dist = sys.maxsize
        min_dist_index = -1

        for l in leader_indices:
            cur_dist = euclidean_distances(X[f], X[l])
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_dist_index = l

        clusters[min_dist_index].append((f, min_dist[0][0]))

    return clusters


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/asama1', methods=['POST', 'GET'])
def asama1():
    if request.method == 'POST':
        url = request.form['url']
        crawler = Crawler(url)
        crawler.crawl()
        word_count = crawler.wordCountWrite()

        return render_template('asama1.html',
                               word_count=word_count,
                               report=crawler.__str__(),
                               is_post=True)
    else:
        return render_template('asama1.html',
                               is_post=False)


@app.route('/asama23', methods=['POST', 'GET'])
def asama23():
    if request.method == 'POST':
        url = request.form['url']
        url2 = request.form['url2']
        crawler = Crawler(url)
        crawler2 = Crawler(url2)
        crawler.crawl()
        crawler2.crawl()
        word_count = crawler.wordCountWrite()
        word_count2 = crawler2.wordCountWrite()
        crawler.keywordWrite()
        crawler2.keywordWrite()
        cluster1 = cluster_docs(5, word_count)
        # cluster_docs(2,word_count2)

        return render_template('asama23.html',
                               word_count=word_count,
                               key_word1=crawler.keyWord,
                               key_word2=crawler2.keyWord,
                               cluster=cluster1,
                               is_post=True)
    else:
        return render_template('asama23.html',
                               is_post=False)


@app.route('/asama4')
def asama4():
    return render_template('asama4.html')


@app.route('/asama5')
def asama5():
    return render_template('asama5.html')


if __name__ == '__main__':
    app.run(debug=True)
