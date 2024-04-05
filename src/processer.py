from helper import DOC_INFO_FILE, INV_IDX_FILE, NUM_DOCS, VOCAB_FILE, VOCAB_SIZE, load_adj_list, load_doc_info, load_inv_idx

import numpy as np
import pandas as pd
import scipy.sparse as sp
import os

def create_vocab() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the vocab

    :returns:
        inv idx
        vocab
    """

    print('Creating vocab ...')
    if os.path.exists(VOCAB_FILE):
        os.remove(VOCAB_FILE)

    inv_idx = load_inv_idx()

    vocab = inv_idx.groupby('term').sum()
    vocab = vocab.sort_values(by='frequency', ascending=False).head(VOCAB_SIZE)

    vocab.to_parquet(VOCAB_FILE, engine='pyarrow')

    print('Finished creating vocab\n')

    return inv_idx, vocab

def reduce_and_sort(inv_idx: pd.DataFrame, vocab: pd.DataFrame) -> None:
    """Load, reduce, and store the inverted index and doc info"""

    print('Reducing and sorting ...')

    terms = set(vocab.index)

    # reduce and sort inverted index
    inv_idx = inv_idx.query('term in @terms').sort_index()
    inv_idx.to_parquet(INV_IDX_FILE, engine='pyarrow')

    print('Finished reducing and sorting inv idx\n')

    # load, reduce, and sort doc labels
    doc_info = load_doc_info()
    doc_info = doc_info.sort_index()
    doc_info.to_parquet(DOC_INFO_FILE, engine='pyarrow')

    print('Finished sorting doc info\n')

    return

def calc_PageRanks() -> None:
    """Calculate the page rank for each document"""

    print('Calculating PageRanks ...')

    doc_info = load_doc_info()
    adj_list = load_adj_list()

    subs = { row['url']: docid for docid, row in doc_info.iterrows() }

    # create transition matrix (prob)
    # https://stackoverflow.com/questions/60894395/quickly-creating-scipy-sparse-matrix-from-adjacency-list
    row, col, data = [], [], []
    for docid, adj_list in adj_list.iterrows():
        out_links = list(adj_list['out_links'])

        # reduce and replace uls w/ ids
        out_links = [ subs[url] for url in out_links if url in subs.keys() ]

        m = len(out_links)
        if m == 0:
            continue

        col.append(out_links)
        data.append([1/m] * m)
        row.append([docid] * m)

    data = np.hstack(data)
    row = np.hstack(row)
    col = np.hstack(col)

    M = sp.coo_matrix((data, (row, col)), (NUM_DOCS, NUM_DOCS))

    # calc matrix eigen vector -> power iteration as found on wikipedia
    b_k = np.random.rand(NUM_DOCS)
    for _ in range(10):
        b_k1 = M.dot(b_k)

        b_k = b_k1 / np.linalg.norm(b_k1)

    # add to and save to doc info
    doc_info['PageRank'] = b_k
    doc_info.to_parquet(DOC_INFO_FILE, engine='pyarrow')

    print('Finished calculating PageRanks\n')

    return
