from helper import DOC_INFO_FILE, INV_IDX_FILE, NUM_DOCS, VOCAB_FILE, VOCAB_SIZE, load_adj_list, load_aliases, load_doc_info, load_inv_idx

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
    aliases  = load_aliases()
    adj_list = load_adj_list()

    subs = { str(row['title']).replace(' ', '_'): docid for docid, row in doc_info.iterrows() }

    aliases = { str(from_title): subs[str(row['to'])] for from_title, row in aliases.iterrows() if str(row['to']) in subs }

    subs |= aliases


    # create transition matrix (prob)
    # https://stackoverflow.com/questions/60894395/quickly-creating-scipy-sparse-matrix-from-adjacency-list
    row, col, data = [], [], []
    for docid, adj_list in adj_list.iterrows():
        out_links = list(adj_list['out_links'])

        # reduce and replace uls w/ ids
        out_links = set([ subs[title] for title in out_links if title in subs.keys() ])

        m = len(out_links)
        if m == 0:
            continue

        col.append(list(out_links))
        data.append([1/m] * m)
        row.append([docid] * m)

    data = np.hstack(data)
    row = np.hstack(row)
    col = np.hstack(col)

    M = sp.coo_matrix((data, (row, col)), (NUM_DOCS, NUM_DOCS))

    # calc matrix eigen vector -> PageRank as found on wikipedia
    MAX_NUM_ITER = 100
    TARGET_ERROR = 1e-6
    d = .85

    # TODO: update formula -> https://medium.com/polo-club-of-data-science/pagerank-algorithm-explained-with-examples-a5e25e2594c9

    M_hat = d * M

    b_k = np.ones(NUM_DOCS)
    b_k /= np.linalg.norm(b_k, 3)
    for _ in range(MAX_NUM_ITER):
        b_old = b_k

        b_k = M_hat.dot(b_k) + ((1 - d) / NUM_DOCS)
        b_k /= np.linalg.norm(b_k, 3)

        if np.linalg.norm(b_old - b_k) < TARGET_ERROR:
            break

    print(f'sum: {b_k.sum()}')
    print(b_k[:5])

    # add to and save to doc info
    doc_info['PageRank'] = b_k
    doc_info.to_parquet(DOC_INFO_FILE, engine='pyarrow')

    print('Finished calculating PageRanks\n')

    return
