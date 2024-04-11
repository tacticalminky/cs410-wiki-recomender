from helper import DOC_INFO_FILE, NUM_DOCS, load_adj_list, load_aliases, load_doc_info

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.preprocessing import normalize as sk_normalize

from tqdm import tqdm

def _normalize(b: np.ndarray, MAX_RANK: int) -> None:
    """Normalize the arrary to the range [0, MAX_RANK]"""

    b_min = b.min()
    b_max = b.max()

    b -= b_min
    b *= MAX_RANK / (b_max - b_min)

    return

def _create_adj_matrix(doc_info: pd.DataFrame) -> sp.csr_matrix:
    """Create the adjacency matrix of the dataset

    :returns: adjacency matrix (M) of the dataset
    """

    aliases  = load_aliases()
    adj_list = load_adj_list()

    subs = { str(row['title']).replace(' ', '_'): docid for docid, row in doc_info.iterrows() }

    aliases = { str(from_title): subs[str(row['to'])] for from_title, row in aliases.iterrows() if str(row['to']) in subs }

    subs |= aliases

    # https://stackoverflow.com/questions/60894395/quickly-creating-scipy-sparse-matrix-from-adjacency-list
    row, col, data = [], [], []
    print('Constructing the adjacency matrix ...')
    for docid, adj_list in tqdm(adj_list.iterrows(), total=NUM_DOCS):
        out_links = list(adj_list['out_links'])

        # reduce and replace uls w/ ids
        out_links = set([ subs[title] for title in out_links if title in subs.keys() ])

        m = len(out_links)
        if m == 0:
            continue

        col.append(list(out_links))
        data.append([1] * m)
        row.append([docid] * m)

    data = np.hstack(data)
    row = np.hstack(row)
    col = np.hstack(col)

    M = sp.coo_matrix((data, (row, col)), (NUM_DOCS, NUM_DOCS)).tocsr()

    print('Finished\n')

    return M

def _calc_HITS_scores(doc_info: pd.DataFrame, M: sp.csr_matrix) -> None:
    """Calculate the HITS score for each document

    Calculated as discused in class w/ normalization between [0,10]
    """

    print('Calculating HITS scores ...')
    NUM_ITER = 1000
    MAX_RANK = 10

    M_T = M.transpose()

    hubs  = np.ones(NUM_DOCS)
    auths = np.ones(NUM_DOCS)
    for _ in tqdm(range(NUM_ITER)):
        hubs  = M.dot(auths)
        auths = M_T.dot(hubs)

        _normalize(auths, MAX_RANK)
        _normalize(hubs,  MAX_RANK)

    # add to and save to doc info
    doc_info['hub_score']  = hubs
    doc_info['auth_score'] = auths
    doc_info.to_parquet(DOC_INFO_FILE, engine='pyarrow')

    print('Finished\n')

    # print('Top auth scores:')
    # print(doc_info.loc[auths.argsort()[::-1][:5]])
    # print()

    # print('Top hub scores:')
    # print(doc_info.loc[hubs.argsort()[::-1][:5]])
    # print()

    return

def _calc_PageRanks(doc_info: pd.DataFrame, M: sp.csr_matrix) -> None:
    """Calculate the PageRank for each document

    Calculated as with help from the following sources:
        - https://en.wikipedia.org/wiki/PageRank
        - https://en.wikipedia.org/wiki/Power_iteration
        - https://medium.com/polo-club-of-data-science/pagerank-algorithm-explained-with-examples-a5e25e2594c9
    """

    print('Calculating PageRanks ...')
    NUM_ITER = 2500
    MAX_RANK = 10

    d = .85

    M_hat: sp.coo_matrix = d * M

    b_k = np.ones(NUM_DOCS) # don't normalize until after first multiplication
    for _ in tqdm(range(NUM_ITER)):
        b_k = M_hat.dot(b_k) + (MAX_RANK * (1 - d))

        _normalize(b_k, MAX_RANK)

    # add to and save to doc info
    doc_info['PageRank'] = b_k
    doc_info.to_parquet(DOC_INFO_FILE, engine='pyarrow')

    print('Finished calculating PageRanks\n')

    # print('Top PageRanks:')
    # print(doc_info.loc[b_k.argsort()[::-1][:5]])
    # print()

    return

def calc_link_ranks() -> None:
    doc_info = load_doc_info()

    M = _create_adj_matrix(doc_info)

    _calc_HITS_scores(doc_info, M)

    # convert adj matrix to transition matrix
    M: sp.csr_matrix = sk_normalize(M, norm='l1')

    _calc_PageRanks(doc_info, M)

    return
