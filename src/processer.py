from helper import DOC_INFO_FILE, INV_IDX_FILE, VOCAB_FILE, VOCAB_SIZE
from helper import load_doc_info, load_inv_idx

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

    vocab.to_csv(VOCAB_FILE, header=False)

    print('Finished')

    return inv_idx, vocab

def reduce_and_sort(inv_idx: pd.DataFrame, vocab: pd.DataFrame) -> None:
    """Load, reduce, and store the inverted index and doc info"""

    print('Reducing and sorting ...')

    terms = set(vocab.index)

    # reduce and sort inverted index
    inv_idx = inv_idx.query('term in @terms').sort_index()
    inv_idx.to_csv(INV_IDX_FILE, header=False)

    print('Finished inv idx')

    # load, reduce, and sort doc labels
    doc_info = load_doc_info()
    doc_info = doc_info.sort_index()
    doc_info.to_csv(DOC_INFO_FILE, header=False)

    print('Finished doc info')

    return

