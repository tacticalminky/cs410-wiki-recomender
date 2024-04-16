from helper import ADJ_LIST_FILE, DOC_INFO_FILE, INV_IDX_FILE

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import signal
import sys
import os

from multiprocessing import Queue

OUTPUT_DIR = '/tmp/wiki_crawler'

class Worker:
    def __init__(self, raw_queue: Queue) -> None:
        """Initialize worker"""

        self.request_session = requests.Session()

        self.inv_idx: np.ndarray | None  = None

        self.docids: list[int]   = []
        self.urls: list[str]     = []
        self.titles: list[str]   = []
        self.doc_lens: list[int] = []

        self.out_links: list[list[str]] = []

        self.writers: dict[str, pq.ParquetWriter] = {}

        self.raw_queue = raw_queue

        self.pid = os.getpid()

        self.doc_info_file = f'{OUTPUT_DIR}/{self.pid}-{DOC_INFO_FILE[7:]}'
        self.inv_idx_file  = f'{OUTPUT_DIR}/{self.pid}-{INV_IDX_FILE[7:]}'
        self.adj_list_file = f'{OUTPUT_DIR}/{self.pid}-{ADJ_LIST_FILE[7:]}'

        for file in (self.doc_info_file, self.inv_idx_file, self.adj_list_file):
            if os.path.exists(file):
                os.remove(file)

        def cleanup(num, frame):
            """Cleanup worker"""

            self.raw_queue.close()

            self.request_session.close()

            if len(self.docids) > 0:
                self.save_data()

            for _, writer in self.writers.items():
                writer.close()

            sys.exit()

        signal.signal(signal.SIGTERM, cleanup)

        return

    def save_data(self) -> None:
        """Save the worker's data to disk"""

        doc_info = pd.DataFrame({'docid': self.docids, 'title': self.titles, 'url': self.urls, 'len': self.doc_lens})
        doc_info = doc_info.astype({'docid': int, 'title': str, 'url': str, 'len': int}).set_index('docid')

        inv_idx = pd.DataFrame(self.inv_idx, columns=['term', 'docid', 'frequency'])
        inv_idx = inv_idx.astype({'term': str, 'docid': int, 'frequency': int}).set_index(['term', 'docid'])

        adj_list = pd.DataFrame({'docid': self.docids, 'out_links': self.out_links})
        adj_list = adj_list.astype({'docid': int}).set_index('docid')

        doc_table = pa.Table.from_pandas(doc_info)
        ii_table  = pa.Table.from_pandas(inv_idx)
        adj_table = pa.Table.from_pandas(adj_list)

        if self.writers.get('doc_info', None) is None:
            self.writers['doc_info'] = pq.ParquetWriter(self.doc_info_file, doc_table.schema)
        self.writers['doc_info'].write_table(doc_table)

        if self.writers.get('ii', None) is None:
            self.writers['ii'] = pq.ParquetWriter(self.inv_idx_file, ii_table.schema)
        self.writers['ii'].write_table(ii_table)

        if self.writers.get('adj_list', None) is None:
            self.writers['adj_list'] = pq.ParquetWriter(self.adj_list_file, adj_table.schema)
        self.writers['adj_list'].write_table(adj_table)

        self.docids.clear()
        self.urls.clear()
        self.titles.clear()
        self.doc_lens.clear()
        self.out_links.clear()

        self.inv_idx = None

        return
