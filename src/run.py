from helper import load_data
from models import prob_ranking, tf_idf_ranking

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

def main() -> None:
    # have user give arg to choose model
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

    # TODO: cli input for query (in while loop)

    # test queries
    for query in ('', 'Computer Science', 'Illinois parallel programming'):
        rankings = rank_query(doc_info, inv_idx, vocab, query)
        print_rankings(doc_info, rankings)

    # TODO: feedback (automatic and direct) -> pseudo counts

    # TODO: save back new models

    return

if __name__ == "__main__":
    main()
    pass

