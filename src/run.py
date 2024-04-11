from helper import load_data, parse_text

import numpy as np
import pandas as pd

TOP_NUM_TO_PRINT = 10

def print_rankings(doc_info: pd.DataFrame, rankings: np.ndarray) -> None:
    """Print the doc info of the top ranked docs

    :param doc_info: the DataFrame of the documents returned in load data
    :param rankings: the ranking of docids sorted in decreadsing relevance
    """

    txt = '\t\t{}) {}\n\t\t\t({})'

    print('\tThe top %d results are:' % TOP_NUM_TO_PRINT)
    for idx in range(0, TOP_NUM_TO_PRINT):
        doc = doc_info.loc[rankings[idx]]
        print(txt.format(idx+1, doc['title'], doc['url']))
    print()

    return

def rank_query(doc_info: pd.DataFrame, inv_idx: pd.DataFrame, vocab: pd.DataFrame, query: str) -> None:
    print('Ranking query: "%s" ...' % query)

    lam = 0.15
    jm_smoothing = (1 - lam) / lam

    col_len = doc_info['len'].sum()

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
    print_rankings(doc_info, rankings)

    print('Finished ranking query')
    print()

    return

def main() -> None:
    # TODO: take in arg to choose model (set default)

    doc_info, inv_idx, vocab = load_data()

    # TODO: cli input for query (in while loop)

    # test queries
    for query in ('', 'Computer Science', 'Illinois parallel programming'):
        rank_query(doc_info, inv_idx, vocab, query)

    # TODO: feedback (automatic and direct) -> pseudo counts

    # save back new models

    return

if __name__ == "__main__":
    main()
    pass

