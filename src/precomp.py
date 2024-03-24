from collections import Counter

import numpy as np
import pandas as pd

from tqdm import tqdm

from nltk.stem import PorterStemmer

VOCAB_SIZE = 20

def main() -> None:
    stemmer = PorterStemmer()

    col_counter = Counter()

    data = pd.read_csv('./data/output.csv', names=['url', 'title', 'content'])

    urls     = []
    titles   = []
    doc_lens = []

    inv_index = None

    for docid, row in tqdm(data.iterrows(), total=len(data)):
        url     = str(row['url'])
        title   = str(row['title'])
        content = str(row['content'])

        doc_counter = Counter()

        words = content.split()
        for word in words:
            word = stemmer.stem(word)
            doc_counter[word] += 1

        col_counter += doc_counter

        urls.append(url)
        titles.append(title)
        doc_lens.append(len(words))

        # term, docid, frequency
        doc_ii = np.array([(str(term), int(docid), cnt) for term, cnt in doc_counter.items()])
        doc_ii = np.reshape(doc_ii, (len(doc_ii), 3))

        if inv_index is None:
            inv_index = doc_ii.copy()
        else:
            inv_index = np.vstack((inv_index, doc_ii))

        pass

    # create vocab (term -> collection frequency)
    vocab = col_counter.most_common(VOCAB_SIZE)
    terms = []
    freqs = []
    for term, freq in vocab:
        terms.append(term)
        freqs.append(freq)

    vocab_data = pd.DataFrame({'term': terms, 'frequency': freqs})
    vocab_data.to_csv('./data/vocab.csv', index=False, header=False)

    # create inverted index (term -> docid -> frequency)
    inverted_index = pd.DataFrame(inv_index, columns=['term', 'docid', 'frequency'])
    inverted_index = inverted_index.query('term in @terms')
    inverted_index.to_csv('./data/inverted_index.csv', index=False, header=False)

    # create doc labels (docid -> url, title, len)
    doc_labels = pd.DataFrame({'docid': range(len(urls)), 'url': urls, 'title': titles, 'len': doc_lens})
    # TODO: filter out docs
    doc_labels.to_csv('./data/doc_labels.csv', index=False, header=False)

    return

if __name__ == "__main__":
    main()
