from helper import NUM_DOCS, ADJ_LIST_FILE, DOC_INFO_FILE, INV_IDX_FILE, parse_text

from sys import flags
if flags.dev_mode:
    import yappi

    LOG_FILE = 'yappi.out'

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import os

from collections import Counter
from bs4 import BeautifulSoup, SoupStrainer
from queue import Queue
from threading import Thread, Lock
from tqdm import tqdm

NUM_TREADS = 6
BATCH_SIZE = 13

MAX_QUEUE_SIZE = NUM_DOCS

BASE_URL = 'https://en.wikipedia.org'

SEED_LINKS = (
    'https://en.wikipedia.org/wiki/University_of_Illinois_Urbana-Champaign',
    'https://en.wikipedia.org/wiki/Computer_science'
)

def _save_data(docids: list[int], urls: list[str], titles: list[str], doc_lens: list[int], out_links: list[list[str]],
               thread_ii: np.ndarray, pbar: tqdm, locks: dict[str, Lock], writers: dict[str, pq.ParquetWriter]) -> None:
    """Save the data to disk"""

    doc_info = pd.DataFrame({'url': urls, 'title': titles, 'len': doc_lens}, index=docids)
    doc_info.index.name = 'docid'

    table = pa.Table.from_pandas(doc_info)
    with locks['doc_info']:
        # doc_info.to_csv(DOC_INFO_FILE, mode='a', header=False, compression='gzip')

        if writers.get('doc_info', None) is None:
            writers['doc_info'] = pq.ParquetWriter(DOC_INFO_FILE, table.schema)
        writers['doc_info'].write_table(table)


    urls.clear()
    titles.clear()
    doc_lens.clear()

    inv_idx = pd.DataFrame(thread_ii, columns=['term', 'docid', 'frequency'])
    inv_idx = inv_idx.set_index(['term', 'docid'])


    table = pa.Table.from_pandas(inv_idx)
    with locks['inv_idx']:
        # inv_idx.to_csv(INV_IDX_FILE, mode='a', index=False, header=False, compression='gzip')

        if writers.get('ii', None) is None:
            writers['ii'] = pq.ParquetWriter(INV_IDX_FILE, table.schema)
        writers['ii'].write_table(table)


    adj_list = pd.DataFrame({'out_links': out_links}, index=docids)
    adj_list.index.name = 'docid'

    table = pa.Table.from_pandas(adj_list)
    with locks['adj_list']:
        if writers.get('adj_list', None) is None:
            writers['adj_list'] = pq.ParquetWriter(ADJ_LIST_FILE, table.schema)
        writers['adj_list'].write_table(table)

        pbar.update(len(docids))

    docids.clear()
    out_links.clear()

    return

def _thread_task(queue: Queue[str], visited: set[str], queue_lock: Lock, data_locks: dict[str, Lock], writers: dict[str, pq.ParquetWriter], pbar: tqdm) -> None:
    """The task of each thread"""

    request_session = requests.Session()
    strainer = SoupStrainer(attrs={'id': ['firstHeading', 'bodyContent']})

    thread_ii = None

    docids: list[int]   = []
    urls: list[str]     = []
    titles: list[str]   = []
    doc_lens: list[int] = []

    list_out_links: list[list[str]] = []

    while True:
        # get a new page
        with queue_lock:
            if len(visited) >= NUM_DOCS:
                break

            link = queue.get()
            while link in visited:
                link = queue.get()

            docid = len(visited)
            visited.add(link)

        # get page and title
        req = request_session.get(link)
        # TODO: check req status (fail, unauth, redirect, etc.)
        # if req.status_code != 200:

        page = BeautifulSoup(req.text, 'lxml', parse_only=strainer)

        title = page.find(id='firstHeading').text

        body = page.find(id='bodyContent')

        # find and queue new links
        out_links: set[str] = set()
        for sub_link in body.find_all('a'):
            if not sub_link.has_attr('href'): continue

            sub_link = str(sub_link['href'])

            sub_link = sub_link.split('#', 1)[0]
            if (not sub_link.startswith('/wiki/')
                or sub_link.startswith('/wiki/File:')
                or sub_link.startswith('/wiki/Help:')
                or sub_link.startswith('/wiki/Special:')
                or sub_link.startswith('/wiki/Template:')
                or sub_link.startswith('/wiki/Template_talk:')):
                continue

            sub_link = BASE_URL + sub_link
            if (sub_link not in visited and sub_link not in out_links):
                out_links.add(sub_link)

        for sub_link in out_links:
            try:
                queue.put_nowait(sub_link)
            except:
                break

        # TODO: save out_links for PageRank

        # build, parse, and count text
        text = ' '.join([ par.text for par in body.find_all('p') ])
        filtered = parse_text(text)

        doc_counter = Counter(filtered)

        # add page data
        docids.append(docid)
        urls.append(link)
        titles.append(title)
        doc_lens.append(len(filtered))
        list_out_links.append(list(out_links))

        # append inv idx
        doc_ii = np.array([(str(term), docid, cnt) for term, cnt in doc_counter.items()])
        doc_ii = np.reshape(doc_ii, (len(doc_ii), 3))

        if thread_ii is None:
            thread_ii = doc_ii.copy()
        else:
            thread_ii = np.vstack((thread_ii, doc_ii))

        # save docs to files every BATCH_SIZE iterations
        if len(urls) >= BATCH_SIZE:
            _save_data(docids, urls, titles, doc_lens, list_out_links, thread_ii, pbar, data_locks, writers)
            thread_ii = None

    # save docs that have yet to be
    if not thread_ii is None:
        _save_data(docids, urls, titles, doc_lens, list_out_links, thread_ii, pbar, data_locks, writers)
        thread_ii = None

    return

def crawl() -> None:
    """Crawl web pages and build an inveted index for them"""

    # del old files
    for file in (ADJ_LIST_FILE, DOC_INFO_FILE, INV_IDX_FILE):
        if os.path.exists(file):
            os.remove(file)

    # init vars for counter
    queue: Queue[str] = Queue(maxsize=MAX_QUEUE_SIZE)
    for link in SEED_LINKS:
        queue.put(link)

    visited = set()

    queue_lock = Lock()

    data_locks: dict[str, Lock] = {}
    for s in ('doc_info', 'inv_idx', 'adj_list'):
        data_locks[s] = Lock()

    writers: dict[str, pq.ParquetWriter] = {}

    if flags.dev_mode:
        print('Running in DEV mode...\n')
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

        yappi.start()

    print('Starting threads to scrap pages:')
    pbar = tqdm(total=NUM_DOCS)

    threads: list[Thread] = []
    for _ in range(NUM_TREADS):
        thread = Thread(target=_thread_task, args=(queue, visited, queue_lock, data_locks, writers, pbar))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    pbar.close()
    print()

    if flags.dev_mode:
        yappi.stop()

        print('Saving file to "%s"\n' % LOG_FILE)

        yappi.get_func_stats().save(LOG_FILE, type='callgrind')

    for _, writer in writers.items():
        writer.close()

    return
