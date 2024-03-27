from helper import MAX_NUM_DOCS, VOCAB_SIZE, DOC_INFO_FILE, INV_IDX_FILE, VOCAB_FILE
from helper import load_inv_idx, parse_text

import numpy as np
import pandas as pd
import requests
import os

from collections import Counter
from threading import Thread, Lock
from queue import Queue

from bs4 import BeautifulSoup
from tqdm import tqdm

NUM_TREADS = 6
MAX_BATCH_SIZE = 25

BASE_URL = 'https://en.wikipedia.org'

SEED_LINKS = (
    'https://en.wikipedia.org/wiki/University_of_Illinois_Urbana-Champaign',
    'https://en.wikipedia.org/wiki/Computer_science'
)

def thread_task(queue: Queue[str], lock: Lock, visited: set[str], pbar: tqdm, col_counter: Counter) -> None:
    # TODO: split into multiple functions

    thread_counter = Counter()
    thread_ii = None

    docids: list[int] = []
    urls: list[str] = []
    titles: list[str] = []
    doc_lens: list[int] = []

    while True:
        # get a new page
        with lock:
            if len(visited) >= MAX_NUM_DOCS:
                break

            link = queue.get()
            while link in visited:
                link = queue.get()

            docid = len(visited)
            visited.add(link)

        # get page and title
        req = requests.get(link)
        # TODO: check req status

        page = BeautifulSoup(req.text, 'html.parser')

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
                sub_link = BASE_URL + sub_link
                if sub_link not in visited and sub_link not in found_links:
                    found_links.add(sub_link)

        for sub_link in found_links:
            queue.put(sub_link)

        # build and parse text
        text = ''
        for par in body.find_all('p'):
            text += ' ' + par.text

        filtered = parse_text(text)

        # count terms
        doc_counter = Counter()
        for term in filtered:
            doc_counter[term] += 1

        thread_counter += doc_counter

        # add page data
        docids.append(docid)
        urls.append(link)
        titles.append(title)
        doc_lens.append(len(filtered))

        # append inv idx
        doc_ii = np.array([(str(term), docid, cnt) for term, cnt in doc_counter.items()])
        doc_ii = np.reshape(doc_ii, (len(doc_ii), 3))

        if thread_ii is None:
            thread_ii = doc_ii.copy()
        else:
            thread_ii = np.vstack((thread_ii, doc_ii))

        # save docs to files every MAX_BATCH_SIZE iterations
        if len(urls) >= MAX_BATCH_SIZE:
            doc_info = pd.DataFrame({'docid': docids, 'url': urls, 'title': titles, 'len': doc_lens})
            inv_idx  = pd.DataFrame(thread_ii, columns=['term', 'docid', 'frequency'])

            with lock:
                doc_info.to_csv(DOC_INFO_FILE, mode='a', index=False, header=False)
                inv_idx.to_csv(INV_IDX_FILE, mode='a', index=False, header=False)

                pbar.update(len(urls))

            thread_ii = None

            docids.clear()
            urls.clear()
            titles.clear()
            doc_lens.clear()

    # save docs that have yet to be
    if len(urls) > 0:
        doc_info = pd.DataFrame({'docid': docids, 'url': urls, 'title': titles, 'len': doc_lens})
        inv_idx  = pd.DataFrame(thread_ii, columns=['term', 'docid', 'frequency'])

        with lock:
            doc_info.to_csv(DOC_INFO_FILE, mode='a', index=False, header=False)
            inv_idx.to_csv(INV_IDX_FILE, mode='a', index=False, header=False)

            pbar.update(len(urls))

    # update collection term counter
    with lock:
        col_counter += thread_counter

    return

def main() -> None:
    # TODO: split into multiple functions

    # TODO: extract into seperat function -> crawl_and_create_inv_idx(col_counter)
    # del old files
    for file in (DOC_INFO_FILE, INV_IDX_FILE, VOCAB_FILE):
        if os.path.exists(file):
            os.remove(file)

    # init vars for counter
    queue = Queue()
    for link in SEED_LINKS:
        queue.put(link)

    lock    = Lock()
    visited = set()
    col_counter = Counter()

    print('Starting threads to scrap pages:')
    pbar = tqdm(total=MAX_NUM_DOCS)
    threads: list[Thread] = []

    for _ in range(NUM_TREADS):
        thread = Thread(target=thread_task, args=(queue, lock, visited, pbar, col_counter))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    queue.queue.clear()
    visited.clear()

    threads.clear()
    pbar.close()

    # TODO: extract to seperate function -> create_vocab(col_counter)
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
    vocab.to_csv(VOCAB_FILE, index=False, header=False)

    # load and reduce inverted index
    inv_idx = load_inv_idx()
    inv_idx = inv_idx.query('term in @terms')
    # TODO: sort
    inv_idx.to_csv(INV_IDX_FILE, header=False)

    # load and reduce doc labels
    # docids = set(inv_idx.index.get_level_values('docid'))
    # doc_info = load_doc_info()
    # doc_info = doc_info.query('docid in @docids') # remove
    # TODO: sort
    # doc_info.to_csv(DOC_INFO_FILE, header=False)

    print('Finished')

    # TODO: PageRank

    return

if __name__ == "__main__":
    main()
    pass
