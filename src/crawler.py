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

def _save_data(docids: list[int], urls: list[str], titles: list[str], doc_lens: list[int],
               out_links: list[list[str]], thread_ii: np.ndarray, pbar: tqdm,
               lock: Lock, writers: dict[str, pq.ParquetWriter]) -> None:
    """Save the data to disk"""

    doc_info = pd.DataFrame({'docid': docids, 'url': urls, 'title': titles, 'len': doc_lens})
    doc_info = doc_info.astype({'docid': int, 'url': str, 'title': str, 'len': int}).set_index('docid')

    inv_idx = pd.DataFrame(thread_ii, columns=['term', 'docid', 'frequency'])
    inv_idx = inv_idx.astype({'term': str, 'docid': int, 'frequency': int}).set_index(['term', 'docid'])

    adj_list = pd.DataFrame({'docid': docids, 'out_links': out_links})
    adj_list = adj_list.astype({'docid': int}).set_index('docid')

    doc_table = pa.Table.from_pandas(doc_info)
    ii_table  = pa.Table.from_pandas(inv_idx)
    adj_table = pa.Table.from_pandas(adj_list)

    with lock:
        if writers.get('doc_info', None) is None:
            writers['doc_info'] = pq.ParquetWriter(DOC_INFO_FILE, doc_table.schema)
        writers['doc_info'].write_table(doc_table)

        if writers.get('ii', None) is None:
            writers['ii'] = pq.ParquetWriter(INV_IDX_FILE, ii_table.schema)
        writers['ii'].write_table(ii_table)

        if writers.get('adj_list', None) is None:
            writers['adj_list'] = pq.ParquetWriter(ADJ_LIST_FILE, adj_table.schema)
        writers['adj_list'].write_table(adj_table)

        pbar.update(len(docids))

    docids.clear()
    urls.clear()
    titles.clear()
    doc_lens.clear()
    out_links.clear()

    return

def _thread_task(queue: Queue[str], visited: set[str], omitted: set[str], queue_lock: Lock,
                 data_lock: Lock, writers: dict[str, pq.ParquetWriter], pbar: tqdm) -> None:
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
            while link in visited or link in omitted:
                link = queue.get()

            docid = len(visited)
            visited.add(link)

        # get html
        req = request_session.get(link)
        # check for undesirable status codes (redirect, not found, etc.)
        while req.status_code != 200:
            # get new url and try again
            with queue_lock:
                print(f'getting new link due to status of {req.status_code}, old link: {link}')
                omitted.add(link)
                visited.remove(link)

                while link in visited or link in omitted:
                    link = queue.get()

                visited.add(link)

            req = request_session.get(link)

        # parse html and get parts
        page = BeautifulSoup(req.text, 'lxml', parse_only=strainer)

        title = page.find(id='firstHeading').text

        body = page.find(id='bodyContent')

        # find outbound links
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
            out_links.add(sub_link)

        # queue new links
        for sub_link in out_links:
            if sub_link not in visited or sub_link not in omitted:
                try:
                    queue.put_nowait(sub_link)
                except:
                    break

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
            _save_data(docids, urls, titles, doc_lens, list_out_links, thread_ii, pbar, data_lock, writers)
            thread_ii = None

    # save docs that have yet to be
    if not thread_ii is None:
        _save_data(docids, urls, titles, doc_lens, list_out_links, thread_ii, pbar, data_lock, writers)
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
    omit    = set()

    queue_lock = Lock()
    data_lock = Lock()

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
        thread = Thread(target=_thread_task, args=(queue, visited, omit, queue_lock, data_lock, writers, pbar))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    pbar.close()
    print()

    if flags.dev_mode:
        yappi.stop()

        print(f'Saving file to "{LOG_FILE}"\n')

        yappi.get_func_stats().save(LOG_FILE, type='callgrind')

    for _, writer in writers.items():
        writer.close()

    return
