from helper import MAX_NUM_DOCS, DOC_INFO_FILE, INV_IDX_FILE
from helper import parse_text

import numpy as np
import pandas as pd
import requests
import os

from collections import Counter
from bs4 import BeautifulSoup
from queue import Queue
from threading import Thread, Lock
from tqdm import tqdm

NUM_TREADS = 6
BATCH_SIZE = 25

MAX_QUEUE_SIZE = MAX_NUM_DOCS

BASE_URL = 'https://en.wikipedia.org'

SEED_LINKS = (
    'https://en.wikipedia.org/wiki/University_of_Illinois_Urbana-Champaign',
    'https://en.wikipedia.org/wiki/Computer_science'
)

def _save_data(docids: list[int], urls: list[str], titles: list[str], doc_lens: list[int],
               thread_ii: np.ndarray | None, pbar: tqdm, lock: Lock) -> None:
    """Save the data to disk"""

    doc_info = pd.DataFrame({'docid': docids, 'url': urls, 'title': titles, 'len': doc_lens})
    inv_idx  = pd.DataFrame(thread_ii, columns=['term', 'docid', 'frequency'])

    with lock:
        doc_info.to_csv(DOC_INFO_FILE, mode='a', index=False, header=False)
        inv_idx.to_csv(INV_IDX_FILE, mode='a', index=False, header=False)

        pbar.update(len(urls))

    docids.clear()
    urls.clear()
    titles.clear()
    doc_lens.clear()

    return

def _thread_task(queue: Queue[str], visited: set[str], queue_lock: Lock, data_lock: Lock, pbar: tqdm) -> None:
    """The task of each thread"""
    thread_ii = None

    docids: list[int]   = []
    urls: list[str]     = []
    titles: list[str]   = []
    doc_lens: list[int] = []

    while True:
        # get a new page
        with queue_lock:
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
                if sub_link not in visited:
                    found_links.add(sub_link)

        for sub_link in found_links:
            try:
                queue.put_nowait(sub_link)
            except:
                break

        # build and parse text
        text = ''
        for par in body.find_all('p'):
            text += ' ' + par.text

        filtered = parse_text(text)

        # count terms
        doc_counter = Counter()
        for term in filtered:
            doc_counter[term] += 1

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

        # save docs to files every BATCH_SIZE iterations
        if len(urls) >= BATCH_SIZE:
            _save_data(docids, urls, titles, doc_lens, thread_ii, pbar, data_lock)
            thread_ii = None

    # save docs that have yet to be
    if len(urls) > 0:
        _save_data(docids, urls, titles, doc_lens, thread_ii, pbar, data_lock)
        thread_ii = None

    return

def crawl() -> None:
    """Crawl web pages and build an inveted index off of them
    """

    # del old files
    for file in (DOC_INFO_FILE, INV_IDX_FILE):
        if os.path.exists(file):
            os.remove(file)

    # init vars for counter
    queue: Queue[str] = Queue(maxsize=MAX_QUEUE_SIZE)
    for link in SEED_LINKS:
        queue.put(link)

    visited = set()

    queue_lock = Lock()
    data_lock  = Lock()

    print('Starting threads to scrap pages:')
    pbar = tqdm(total=MAX_NUM_DOCS)

    threads: list[Thread] = []
    for _ in range(NUM_TREADS):
        thread = Thread(target=_thread_task, args=(queue, visited, queue_lock, data_lock, pbar))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    pbar.close()

    return
