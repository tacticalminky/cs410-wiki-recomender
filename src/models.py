from helper import NUM_DOCS, parse_text

import numpy as np
import pandas as pd

def prob_ranking(doc_info: pd.DataFrame, inv_idx: pd.DataFrame, vocab: pd.DataFrame, query: str) -> np.ndarray:
    """Rank the query using a probabilistic model

    :param doc_info:    DataFrame of document info
    :param inv_idx:     DataFrame of the inverted index
    :param vocab:       DataFrame of the vocab
    :param query:       Query to be ranked with the model

    :returns:   The document indecies in decreasing order of ranking
    """

    print('Ranking query: "%s" ...' % query)

    # set smoothing params
    lam = 0.15
    jm_smoothing = (1 - lam) / lam

    col_len = doc_info['len'].sum()

    # init doc relivance
    # doc_rel = np.zeros(NUM_DOCS)
    doc_rel = np.array(doc_info['PageRank'] + 2*doc_info['hub_score'] + doc_info['auth_score'])
    doc_rel /= 4

    filtered = parse_text(query)
    for term in filtered:
        if term not in vocab.index:
            continue

        col_prob = vocab.loc[term].iloc[0] / col_len

        term_data = inv_idx.loc[term]
        doc_ids = set(term_data.index)  # docs containing term
        for id in doc_ids:
            doc_len = doc_info.loc[id]['len']
            doc_cnt = term_data.loc[id].iloc[0]

            doc_prob = doc_cnt / doc_len

            doc_rel[id] += np.log(1 + jm_smoothing * (doc_prob / col_prob))

    rankings = doc_rel.argsort()[::-1]
    # print(doc_rel[rankings[:10]])

    print('Finished ranking query\n')

    return rankings

def tf_idf_ranking(doc_info: pd.DataFrame, inv_idx: pd.DataFrame, vocab: pd.DataFrame, query: str) -> np.ndarray:
    """Rank the query using a TF-IDF model

    :param doc_info:    DataFrame of document info
    :param inv_idx:     DataFrame of the inverted index
    :param vocab:       DataFrame of the vocab
    :param query:       Query to be ranked with the model

    :returns:   The document indecies in decreasing order of ranking
    """

    print('Ranking query: "%s" ...' % query)

    # set smoothing params
    k: int = 10
    b: float = 0.25

    avg_doc_len = doc_info['len'].mean()

    # init doc relivance with link rankings
    # doc_rel = np.zeros(NUM_DOCS)
    doc_rel = np.array(doc_info['PageRank'] + 2*doc_info['hub_score'] + doc_info['auth_score'])
    doc_rel /= 4

    filtered = parse_text(query)
    for term in filtered:
        if term not in vocab.index:
            continue

        term_data = inv_idx.loc[term]
        doc_ids = set(term_data.index)  # docs containing term
        for id in doc_ids:
            doc_cnt = term_data.loc[id].iloc[0]
            doc_len = doc_info.loc[id]['len']

            numerator = (k + 1) * doc_cnt
            divisor = doc_cnt + k * (1 - b + b * (doc_len / avg_doc_len))

            idf = np.log((NUM_DOCS + 1) / len(doc_ids))

            doc_rel[id] += (numerator / divisor) * idf

    rankings = doc_rel.argsort()[::-1]
    # print(doc_rel[rankings[:10]])

    print('Finished ranking query\n')

    return rankings
