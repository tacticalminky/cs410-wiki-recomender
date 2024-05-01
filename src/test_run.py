from helper import load_data, parse_text
from models import prob_ranking, tf_idf_ranking

import numpy as np
import pandas as pd

TOP_NUM_TO_PRINT = 10

def _print_rankings(doc_info: pd.DataFrame, rankings: np.ndarray) -> None:
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

def _update_with_feedback(doc_info: pd.DataFrame, inv_idx: pd.DataFrame, vocab: pd.DataFrame, query: str, docids: list[int]) -> None:
    """Updates the documents with the provided feedback"""

    print('Updating with feedback...')

    filtered = parse_text(query)
    for term in filtered:
        if term not in vocab.index:
            continue

        psuedo_term_cnt = 2
        for id in docids:
            inv_idx.loc[term, id] += psuedo_term_cnt
            doc_info.loc[id, 'len'] += psuedo_term_cnt

        vocab.loc[term] += psuedo_term_cnt * len(docids)

    print('Finished\n')

    return

def main() -> None:
    # load data
    doc_info, inv_idx, vocab = load_data()

    test_queries = ('', 'Computer Science', 'Illinois parallel programming')

    # testing queries
    print('Testing queries w/ TF-IDF\n')
    for query in test_queries:
        rankings = tf_idf_ranking(doc_info, inv_idx, vocab, query)
        _print_rankings(doc_info, rankings)

    print('\nTesting queries w/ Prob Ranking\n')
    for query in test_queries:
        rankings = prob_ranking(doc_info, inv_idx, vocab, query)
        _print_rankings(doc_info, rankings)

    return

if __name__ == "__main__":
    main()
    pass

