import numpy as np
import pandas as pd

from tqdm import tqdm

from nltk.stem import PorterStemmer

TOP_NUM_TO_PRINT = 10

DOC_INFO_FILE   = './data/doc_info.csv'
INV_IDX_FILE    = './data/inv_idx.csv'
VOCAB_FILE      = './data/vocab.csv'

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    doc_info = pd.read_csv(DOC_INFO_FILE,
                           names=['docid', 'url', 'title', 'len'],
                           index_col='docid')

    inv_idx  = pd.read_csv(INV_IDX_FILE,
                           names=['term', 'docid', 'frequency'],
                           index_col=['term', 'docid'])

    vocab    = pd.read_csv(VOCAB_FILE,
                           names=['term', 'frequency'],
                           index_col='term')

    return (doc_info, inv_idx, vocab)

def print_rankings(doc_info: pd.DataFrame, rankings: np.ndarray) -> None:
    txt = '\t\t{}) {}\n\t\t\t({})'

    print('\tThe top %d results are:' % TOP_NUM_TO_PRINT)
    for idx in range(0, TOP_NUM_TO_PRINT):
        doc = doc_info.loc[rankings[idx]]
        print(txt.format(idx+1, doc['title'], doc['url']))
    print()

    return

def rank_query(doc_info: pd.DataFrame, inv_idx: pd.DataFrame, vocab: pd.DataFrame, query: str) -> None:
    print('Ranking query: "%s" ...' % query)

    # TODO: cleanup query (same as doc contents)

    stemmer = PorterStemmer()

    lam = 0.15
    jm_smoothing = (1 - lam) / lam

    col_len  = doc_info['len'].sum()

    doc_rel = np.zeros(len(doc_info))

    for word in query.split():
        word = stemmer.stem(word.lower())
        if word not in vocab.index:
            continue

        col_prob = vocab.loc[word].iloc[0] / col_len

        word_data = inv_idx.loc[word]
        doc_ids = set(word_data.index)      # docs containing word
        for id in tqdm(doc_ids):
            doc_len = doc_info.loc[id]['len']
            doc_cnt = word_data.loc[id].iloc[0]

            doc_prob = doc_cnt / doc_len

            doc_rel[id] += np.log(1 + jm_smoothing * (doc_prob / col_prob))

    rankings = doc_rel.argsort()[::-1]
    print_rankings(doc_info, rankings)

    print('Finished ranking query')
    print()

    return

def main() -> None:
    doc_info, inv_idx, vocab = load_data()

    for query in ('Illinois teacher of computer science', 'Interquartile range of dogs'):
        rank_query(doc_info, inv_idx, vocab, query)

    return

if __name__ == "__main__":
    main()
    pass

