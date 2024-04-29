import re
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
    # have user choose model
    options = ('0: TF-IDF', '1: Probabilistic')
    txt = 'Which model would you like to use?'
    for option in options:
        txt += f'\n\t{option}'

    model = input(f'{txt}\nPlease enter the number: ').strip()
    print()

    try:
        model = int(model)
    except ValueError:
        print(f'{model} was not an integer.\nPlease select one of the above options.')
        exit()

    match model:
        case 0:
            print('You chose a TF-IDF model\n')
            rank_query = tf_idf_ranking
        case 1:
            print('You chose a probabilistic model\n')
            rank_query = prob_ranking
        case _:
            print(f'{model} was not an opiton.\nPlease select one of the above options.')
            exit()

    # load data
    doc_info, inv_idx, vocab = load_data()

    # cli input for query
    while True:
        query = input('\nPlease enter in a query (or "exit" to exit):\n').strip()
        if query == 'exit':
            break

        print()

        rankings = rank_query(doc_info, inv_idx, vocab, query)
        _print_rankings(doc_info, rankings)

        rel_docs = input('\nWhich docs were relevant?\nPlease enter the numbers seperated with spaces and/or commas:\n').strip()
        rel_docs = re.sub(r',|\s+', ' ', rel_docs)
        rel_docs = rel_docs.split()

        print()

        failed = False
        docids = list()
        for num in rel_docs:
            try:
                idx = int(num) - 1
            except ValueError:
                print(f'{num} was not a number, failed to update ranks')
                failed = True
                break

            if idx >= TOP_NUM_TO_PRINT:
                print(f'{num} is outside of the returned documents, failed to update')
                failed = True
                break

            docids.append(rankings[idx])

        if not failed and len(docids) > 0:
            _update_with_feedback(doc_info, inv_idx, vocab, query, docids)

    print('Exiting ...')

    # test queries
    # for query in ('', 'Computer Science', 'Illinois parallel programming'):
    #     rankings = rank_query(doc_info, inv_idx, vocab, query)
    #     print_rankings(doc_info, rankings)

    return

if __name__ == "__main__":
    main()
    pass

