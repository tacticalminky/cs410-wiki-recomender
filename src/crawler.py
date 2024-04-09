from helper import ALIAS_FILE, NUM_DOCS, ADJ_LIST_FILE, DOC_INFO_FILE, INV_IDX_FILE, parse_text

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
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from queue import Queue
from threading import Thread, Lock
from tqdm import tqdm

# disable specific warning form bs4
from warnings import filterwarnings
filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

NUM_TREADS = 6
BATCH_SIZE = 13

MAX_QUEUE_SIZE = NUM_DOCS

API_URL = 'https://en.wikipedia.org/api/rest_v1/page'

SEEDS = (
    'University_of_Illinois_Urbana-Champaign',
    'Computer_science'
)

def _save_data(docids: list[int], titles: list[str], urls: list[str], doc_lens: list[int],
               out_links: list[list[str]], aliases: dict[str, str], thread_ii: np.ndarray, pbar: tqdm,
               lock: Lock, writers: dict[str, pq.ParquetWriter]) -> None:
    """Save the data to disk"""

    doc_info = pd.DataFrame({'docid': docids, 'title': titles, 'url': urls, 'len': doc_lens})
    doc_info = doc_info.astype({'docid': int, 'title': str, 'url': str, 'len': int}).set_index('docid')

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

    if len(aliases) > 0:
        aliases_df = pd.DataFrame({'from': aliases.keys(), 'to': aliases.values()})
        aliases_df = aliases_df.astype({'from': str, 'to': str}).set_index('from')

        aliases_table = pa.Table.from_pandas(aliases_df)

        with lock:
            if writers.get('aliases', None) is None:
                writers['aliases'] = pq.ParquetWriter(ALIAS_FILE, aliases_table.schema)
            writers['aliases'].write_table(aliases_table)

    aliases.clear()

    return

def _thread_task(queue: Queue[str], visited: set[str], aliased: set[str], omitted: set[str], queue_lock: Lock,
                 data_lock: Lock, writers: dict[str, pq.ParquetWriter], pbar: tqdm) -> None:
    """The task of each thread"""

    request_session = requests.Session()

    thread_ii = None

    docids: list[int]   = []
    urls: list[str]     = []
    titles: list[str]   = []
    doc_lens: list[int] = []

    thread_out_links: list[list[str]] = []

    aliases: dict[str, str] = {}

    while True:
        # get a new page
        with queue_lock:
            if len(visited) >= NUM_DOCS:
                break

            docid = len(visited)

            title = queue.get()
            while title in visited or title in aliased or title in omitted:
                title = queue.get()

            visited.add(title)

        # get the summary of a page and filter out aliases
        summary = None
        while summary is None or title != summary['titles']['canonical']:
            if summary is not None:
                #print(f'{title} was an alias!')
                new_title = summary['titles']['canonical']

                aliases[title] = new_title

                with queue_lock:
                    aliased.add(title)
                    visited.remove(title)

                    if new_title not in visited:
                        visited.add(new_title)
                        title = new_title
                        break

                    title = queue.get()
                    while title in visited or title in aliased or title in omitted:
                        title = queue.get()

                    visited.add(title)

            req = request_session.get(f'{API_URL}/summary/{title}', timeout=1)
            while not req.ok:
                #print(f'getting new title due to status of {req.status_code}, old title: {title}')
                with queue_lock:
                    omitted.add(title)
                    visited.remove(title)

                    title = queue.get()
                    while title in visited or title in aliased or title in omitted:
                        title = queue.get()

                    visited.add(title)

                req = request_session.get(f'{API_URL}/summary/{title}', timeout=1)

            summary = req.json()

        # get the html of the page
        req = request_session.get(f'{API_URL}/html/{title}')
        # TODO: error checking (another while loop?)

        page = BeautifulSoup(req.text, 'lxml')

        # find outbound links
        out_links: set[str] = set()
        for sub_link in page.find_all('a'):
            if not sub_link.has_attr('href'): continue

            # get the link and remove any markers
            sub_link = str(sub_link['href']).split('#', 1)[0]

            # filter out bad links
            if (not sub_link.startswith('./')
                or sub_link.startswith('./File:')
                or sub_link.startswith('./Help:')
                or sub_link.startswith('./Special:')
                or sub_link.startswith('./Template:')
                or sub_link.startswith('./Template_talk:')
                or sub_link.startswith('./Wikipedia:')):
                continue

            out_links.add(sub_link[2:])

        # queue new titles
        for sub_link in out_links:
            if sub_link not in visited or sub_link not in aliased or sub_link not in omitted:
                try:
                    queue.put_nowait(sub_link)
                except:
                    break

        # build, parse, and count text
        text = ' '.join([ par.text for par in page.find_all('p') ])
        filtered = parse_text(text)

        doc_counter = Counter(filtered)

        # add page data
        docids.append(docid)
        titles.append(summary['title'])
        urls.append(summary['content_urls']['desktop']['page'])
        doc_lens.append(len(filtered))
        thread_out_links.append(list(out_links))

        # append inv idx
        doc_ii = np.array([(str(term), docid, cnt) for term, cnt in doc_counter.items()])
        doc_ii = np.reshape(doc_ii, (len(doc_ii), 3))

        if thread_ii is None:
            thread_ii = doc_ii.copy()
        else:
            thread_ii = np.vstack((thread_ii, doc_ii))

        # save docs to files every BATCH_SIZE iterations
        if len(urls) >= BATCH_SIZE:
            _save_data(docids, titles, urls, doc_lens, thread_out_links, aliases, thread_ii, pbar, data_lock, writers)
            thread_ii = None

    # save docs that have yet to be
    if not thread_ii is None:
        _save_data(docids, titles, urls, doc_lens, thread_out_links, aliases, thread_ii, pbar, data_lock, writers)
        thread_ii = None

    return

def crawl() -> None:
    """Crawl web pages and build an inveted index for them"""

    # del old files
    for file in (ALIAS_FILE, ADJ_LIST_FILE, DOC_INFO_FILE, INV_IDX_FILE):
        if os.path.exists(file):
            os.remove(file)

    # init vars for counter
    queue: Queue[str] = Queue(maxsize=MAX_QUEUE_SIZE)
    for title in SEEDS:
        queue.put(title)

    visited = set()
    aliased = set()
    omitted = set()

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
        thread = Thread(target=_thread_task, args=(queue, visited, aliased, omitted, queue_lock, data_lock, writers, pbar))
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

    print(f'\tThere were {len(aliased)} aliased pages and {len(omitted)} omitted pages\n')

    return
