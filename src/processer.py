from helper import DOC_INFO_FILE, INV_IDX_FILE, VOCAB_FILE, VOCAB_SIZE, load_adj_list, load_doc_info, load_inv_idx

import pandas as pd
import os

def create_vocab() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the vocab

    :returns:
        inv idx
        vocab
    """

    inv_idx = load_inv_idx()
    print()

    print('Creating vocab ...')
    if os.path.exists(VOCAB_FILE):
        os.remove(VOCAB_FILE)

    vocab = inv_idx.groupby('term').sum()
    vocab = vocab.sort_values(by='frequency', ascending=False).head(VOCAB_SIZE)

    vocab.to_parquet(VOCAB_FILE, engine='pyarrow')

    print('Finished\n')

    return inv_idx, vocab

def reduce_and_sort(inv_idx: pd.DataFrame, vocab: pd.DataFrame) -> None:
    """Load, reduce, and store the inverted index and doc info"""

    print('Reducing and sorting ...')

    terms = set(vocab.index)

    # reduce and sort inverted index
    inv_idx = inv_idx.query('term in @terms').sort_index()
    inv_idx.to_parquet(INV_IDX_FILE, engine='pyarrow')

    print('Finished inv idx\n')

    # load, reduce, and sort doc labels
    doc_info = load_doc_info()
    doc_info = doc_info.sort_index()
    doc_info.to_parquet(DOC_INFO_FILE, engine='pyarrow')

    print('Finished doc info\n')

    return

def calc_PageRanks() -> None:
    """Calculate the page rank for each document"""

    print('Calculating PageRanks ...')

    doc_info = load_doc_info()
    adj_list = load_adj_list()

    print(adj_list.head())

    #adj_list = adj_list.sort_index()

    subs = { row['url']: docid for docid, row in doc_info.iterrows() }

    # TODO: reduce and replace w/ ids
    # TODO: create transition matrix (prob)

    for docid, adj_list in adj_list.iterrows():
        print(adj_list['out_links'][0])
        return

    # TODO: calc matrix eigen vector

    # TODO: add to and save to doc info

    return
