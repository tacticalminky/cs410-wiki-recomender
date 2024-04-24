from helper import ALIAS_FILE, NUM_DOCS, ADJ_LIST_FILE, DOC_INFO_FILE, INV_IDX_FILE, parse_text
from crawlerWorker import OUTPUT_DIR, Worker

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import os

from collections import Counter
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from multiprocessing import Queue, active_children, set_start_method
from multiprocessing.pool import AsyncResult, Pool
from queue import Queue as ThreadQueue
from threading import Lock, Thread
from tqdm import tqdm

# disable specific warning form bs4
from warnings import filterwarnings
filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

set_start_method('spawn', force=True)

### Declare constants
NUM_ORG_THREADS = 2     # Num threads in main process to queue pages for workers
NUM_WORKERS     = 12    # Num processes to do work

BATCH_SIZE  = 250       # How often to save file

API_URL = 'https://en.wikipedia.org/api/rest_v1/page'

SEEDS = (
    'University_of_Illinois_Urbana-Champaign',
    'Computer_science'
)

NUM_RAND_SEEDS = 2

MAX_RAW_QUEUE_SIZE = NUM_DOCS
MAX_READY_QUEUE_SIZE = 3 * NUM_WORKERS


### Declare worker class
def _init_worker(raw_queue: Queue) -> None:
    """Initialize the worker thread"""

    global _worker
    _worker = Worker(raw_queue)

    return


def _worker_task(docid: int, slug: str, title: str, url: str) -> None:
    """The task of each worker process"""

    # get the html of the page
    res = _worker.request_session.get(f'{API_URL}/html/{slug}')
    # TODO: error checking
    # if not res.ok:

    page = BeautifulSoup(res.text, 'lxml')

    # find outbound links
    out_links: set[str] = set()
    for sub_link in page.find_all('a'):
        if not sub_link.has_attr('href'): continue

        # get the link and remove any markers
        sub_link = str(sub_link['href']).split('#', 1)[0]
        sub_link = sub_link.split('?', 1)[0]

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
        try:
            _worker.raw_queue.put_nowait(sub_link)
        except:
            break

    # build, parse, and count text
    text = ' '.join([ par.text for par in page.find_all('p') ])
    filtered = parse_text(text)

    doc_counter = Counter(filtered)

    # add page data
    _worker.docids.append(docid)
    _worker.titles.append(title)
    _worker.urls.append(url)
    _worker.doc_lens.append(len(filtered))
    _worker.out_links.append(list(out_links))

    # append inv idx
    doc_ii = np.array([(str(term), docid, cnt) for term, cnt in doc_counter.items()])
    doc_ii = np.reshape(doc_ii, (len(doc_ii), 3))

    if _worker.inv_idx is None:
        _worker.inv_idx = doc_ii.copy()
    else:
        _worker.inv_idx = np.vstack((_worker.inv_idx, doc_ii))

    # save docs to files every BATCH_SIZE iterations
    if len(_worker.urls) >= BATCH_SIZE:
        _worker.save_data()

    return

def _prepare_tasks(raw_queue: Queue, ready_queue: ThreadQueue,
                   visited: set[str], aliased: set[str], omitted: set[str],
                   lock: Lock) -> None:
    """The job for the organizer thread"""

    request_session = requests.Session()

    curr_aliases: dict[str, str] = {}

    global writer
    writer = None

    def save_aliases() -> None:
        if len(curr_aliases) == 0:
            return

        aliases_df = pd.DataFrame({'from': curr_aliases.keys(), 'to': curr_aliases.values()})
        aliases_df = aliases_df.astype({'from': str, 'to': str}).set_index('from')

        aliases_table = pa.Table.from_pandas(aliases_df)

        global writer
        with lock:
            if writer is None:
                writer = pq.ParquetWriter(ALIAS_FILE, aliases_table.schema)
            writer.write_table(aliases_table)

        curr_aliases.clear()

        return

    while True:
        with lock:
            if len(visited) >= NUM_DOCS:
                break

            docid = len(visited)

            slug = raw_queue.get()
            while slug in visited or slug in aliased or slug in omitted:
                slug = raw_queue.get()

            visited.add(slug)

        # get the summary of a page and filter out aliases
        summary = None
        while slug is None or summary is None or slug != summary['titles']['canonical']:
            if slug is not None and summary is not None:
                new_slug = summary['titles']['canonical']

                curr_aliases[slug] = new_slug

                with lock:
                    aliased.add(slug)
                    visited.remove(slug)

                    if new_slug not in visited:
                        visited.add(new_slug)
                        slug = new_slug

                        break

                    slug = None

            # get new slug
            if slug is None:
                with lock:
                    slug = raw_queue.get()
                    while slug in visited or slug in aliased or slug in omitted:
                        slug = raw_queue.get()

                    visited.add(slug)

            try:
                res = request_session.get(f'{API_URL}/summary/{slug}', timeout=(1,3))
            except requests.Timeout:
                with lock:
                    omitted.add(slug)
                    visited.remove(slug)

                slug = None
                continue

            if not res.ok:
                with lock:
                    omitted.add(slug)
                    visited.remove(slug)

                slug = None
                continue

            summary = res.json()

        # add to queue
        task = (docid, slug, summary['title'], summary['content_urls']['desktop']['page'])
        ready_queue.put(task)

        # save curr_aliases ever so often
        if len(curr_aliases) >= BATCH_SIZE:
            save_aliases()

    # cleanup
    if len(curr_aliases) > 0:
        save_aliases()

    return

def start_crawler() -> None:
    """Start crawling web pages and building inverted index"""

    # del old files
    for file in (ALIAS_FILE, ADJ_LIST_FILE, DOC_INFO_FILE, INV_IDX_FILE):
        if os.path.exists(file):
            os.remove(file)

    # create output path for the workers
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


    # init queues
    raw_queue: Queue[str] = Queue()
    for title in SEEDS:
        raw_queue.put(title)

    # add random seeds
    print('Adding some random seed pages...')
    for i in range(NUM_RAND_SEEDS):
        res = requests.get(f'{API_URL}/random/summary', timeout=1)

        summary = res.json()
        title = summary['titles']['canonical']

        print(f'\t{i+1}) {title}')

        raw_queue.put(title)

    print()

    ready_queue = ThreadQueue(maxsize=MAX_READY_QUEUE_SIZE)
    results: ThreadQueue[AsyncResult] = ThreadQueue(maxsize=NUM_DOCS)

    # start organizer
    print(f'Starting {NUM_ORG_THREADS} organizer threads...')
    visited = set()
    aliased = set()
    omitted = set()

    lock = Lock()

    organizers = list()
    for _ in range(NUM_ORG_THREADS):
        task_organizer = Thread(target=_prepare_tasks, args=(raw_queue, ready_queue, visited, aliased, omitted, lock))
        task_organizer.start()
        organizers.append(task_organizer)

    # create process pool
    print(f'Starting {NUM_WORKERS} workers...')
    worker_pool = Pool(processes=NUM_WORKERS, initializer=_init_worker, initargs=(raw_queue,))

    # collect pids for joining files
    pids = set()
    children = active_children()
    for child in children:
        pids.add(child.pid)

    print('\nScrapping pages:')

    # create callback for progress bar and dynamic assigning
    pbar = tqdm(total=NUM_DOCS)
    def task_callback(_) -> None:
        pbar.update(1)

        try:
            task = ready_queue.get(timeout=.5)
            results.put(worker_pool.apply_async(_worker_task, args=task, callback=task_callback))
        except:
            return

        return

    # assign first few docs
    for _ in range(min(NUM_DOCS, 3 * NUM_WORKERS)):
        task = ready_queue.get()
        results.put(worker_pool.apply_async(_worker_task, args=task, callback=task_callback))

    # wait for organizer to finish creating tasks
    for task_organizer in organizers:
        task_organizer.join()

    global writer
    if writer is not None:
        writer.close()

    # add remainder of tasks
    while not ready_queue.empty():
        try:
            task = ready_queue.get_nowait()
            results.put(worker_pool.apply_async(_worker_task, args=task, callback=task_callback))
        except:
            break

    # get each task -> won't join otherwise
    for _ in range(NUM_DOCS):
        res = results.get()
        res.get()

    # close and join pool
    worker_pool.terminate()
    worker_pool.join()

    pbar.close()
    print()

    print(f'\tThere were {len(aliased)} aliased pages and {len(omitted)} omitted pages\n')

    print('Joining files from each worker ...')
    for file in (ADJ_LIST_FILE, DOC_INFO_FILE, INV_IDX_FILE):
        data = list()
        for pid in pids:
            data.append(pd.read_parquet(f'{OUTPUT_DIR}/{pid}-{file[7:]}', engine='pyarrow'))

        pd.concat(data).to_parquet(file, engine='pyarrow')

    print('Finished\n')

    return
