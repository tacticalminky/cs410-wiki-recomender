import nltk
import numpy as np
import pandas as pd
import requests
import re
import os

from collections import Counter
from threading import Thread, Lock
from queue import Queue

from bs4 import BeautifulSoup
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

NUM_TREADS = 6

MAX_NUM_ITERATIONS = int(10e3)
MAX_BATCH_SIZE = 25

VOCAB_SIZE = 500

base_url = 'https://en.wikipedia.org'

seed_links = (
    'https://en.wikipedia.org/wiki/University_of_Illinois_Urbana-Champaign',
    'https://en.wikipedia.org/wiki/Computer_science'
)

doc_labels_file = './data/doc_labels.csv'
inv_idx_file    = './data/inv_idx.csv'
vocab_file      = './data/vocab.csv'

def thread_task(queue: Queue[str], lock: Lock, visited: set[str], pbar: tqdm, col_counter: Counter) -> None:
    stemmer = PorterStemmer()

    thread_counter = Counter()
    thread_ii = None

    docids: list[int] = []
    urls: list[str] = []
    titles: list[str] = []
    doc_lens: list[int] = []

    while True:
        # get a new page
        with lock:
            if len(visited) >= MAX_NUM_ITERATIONS:
                break

            link = queue.get()
            while link in visited:
                link = queue.get()

            docid = len(visited)
            visited.add(link)

        # get page and title
        html = requests.get(link).text
        page = BeautifulSoup(html, 'html.parser')

        title = page.find(id='firstHeading').text

        body = page.find(id='bodyContent')

        # find and queue new links
        found_links = set()
        for sub_link in body.find_all('a'):
            if not sub_link.has_attr('href'): continue

            sub_link = sub_link['href']

            if (sub_link.startswith('/wiki/File:')
                or sub_link.startswith('/wiki/Help:')
                or sub_link.startswith('/wiki/Special:')
                or sub_link.startswith('/wiki/Template:')
                or sub_link.startswith('/wiki/Template_talk:')):
                continue

            if sub_link.startswith('/wiki/'):
                sub_link = base_url + sub_link
                if sub_link not in visited and sub_link not in found_links:
                    found_links.add(sub_link)

        for sub_link in found_links:
            queue.put(sub_link)

        # parse text
        text = ''
        for par in body.find_all('p'):
            text += ' ' + par.text

        # remove new line chars, non-alphanumeric chars, and nums
        text = re.sub(r'\n|\r', ' ', text)
        text = re.sub(r"'[\w]*|[^\w\s]+", ' ', text)
        text = re.sub(r'\d+[\w]*', ' ', text)

        # remove extra whitespace and strip
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        text = text.lower()

        # tokenize and remove stopwords
        doc_counter = Counter()
        word_tokens = word_tokenize(text)
        filtered = [ stemmer.stem(w) for w in word_tokens if not w in stop_words ]
        for word in filtered:
            doc_counter[word] += 1

        thread_counter += doc_counter

        # add page data
        docids.append(docid)
        urls.append(link)
        titles.append(title)
        doc_lens.append(len(filtered))

        # append ii
        doc_ii = np.array([(str(term), docid, cnt) for term, cnt in doc_counter.items()])
        doc_ii = np.reshape(doc_ii, (len(doc_ii), 3))

        if thread_ii is None:
            thread_ii = doc_ii.copy()
        else:
            thread_ii = np.vstack((thread_ii, doc_ii))

        # save to files every MAX_BATCH_SIZE iterations
        if len(urls) >= MAX_BATCH_SIZE:
            doc_labels = pd.DataFrame({'docid': docids, 'url': urls, 'title': titles, 'len': doc_lens})
            inv_idx = pd.DataFrame(thread_ii, columns=['term', 'docid', 'frequency'])

            with lock:
                doc_labels.to_csv(doc_labels_file, mode='a', index=False, header=False)
                inv_idx.to_csv(inv_idx_file, mode='a', index=False, header=False)

                pbar.update(len(urls))

            thread_ii = None

            docids.clear()
            urls.clear()
            titles.clear()
            doc_lens.clear()

    # output contents
    if len(urls) > 0:
        doc_labels = pd.DataFrame({'docid': docids, 'url': urls, 'title': titles, 'len': doc_lens})
        inv_idx = pd.DataFrame(thread_ii, columns=['term', 'docid', 'frequency'])

        with lock:
            doc_labels.to_csv(doc_labels_file, mode='a', index=False, header=False)
            inv_idx.to_csv(inv_idx_file, mode='a', index=False, header=False)

            pbar.update(len(urls))

    with lock:
        col_counter += thread_counter

    return

def main() -> None:
    for file in (doc_labels_file, inv_idx_file, vocab_file):
        if os.path.exists(file):
            os.remove(file)

    queue = Queue()
    for link in seed_links:
        queue.put(link)

    lock    = Lock()
    visited = set()
    col_counter = Counter()

    threads: list[Thread] = []

    print('Starting threads to scrap pages:')
    pbar = tqdm(total=MAX_NUM_ITERATIONS)
    for _ in range(NUM_TREADS):
        thread = Thread(target=thread_task, args=(queue, lock, visited, pbar, col_counter))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    pbar.close()

    print()
    print('Creating vocab and reducing inverted index')
    # create vocab
    vocab_counter = col_counter.most_common(VOCAB_SIZE)
    terms = []
    freqs = []
    for term, freq in vocab_counter:
        terms.append(term)
        freqs.append(freq)

    vocab = pd.DataFrame({'term': terms, 'frequency': freqs})
    vocab.to_csv(vocab_file, index=False, header=False)

    # load and reduce inverted index
    inv_idx = pd.read_csv(inv_idx_file, names=['term', 'docid', 'frequency'], index_col=['term', 'docid'])
    inv_idx = inv_idx.query('term in @terms')
    inv_idx.to_csv(inv_idx_file, header=False)

    # load and reduce doc labels
    docids = set(inv_idx.index.get_level_values('docid'))
    doc_labels = pd.read_csv(doc_labels_file, names=['docid', 'url', 'title', 'len'], index_col=['docid'])
    doc_labels = doc_labels.query('docid in @docids')
    doc_labels.to_csv(doc_labels_file, header=False)

    print('Finished')

    return

if __name__ == "__main__":
    main()
    pass
