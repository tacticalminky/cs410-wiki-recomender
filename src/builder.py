from crawler import crawl
from processer import calc_PageRanks, create_vocab, reduce_and_sort

from time import time

def main() -> None:
    crawl()

    inv_idx, vocab = create_vocab()

    reduce_and_sort(inv_idx, vocab)

    del inv_idx, vocab

    t0 = time()
    calc_PageRanks()
    t1 = time()

    print(t1-t0)

    return

if __name__ == "__main__":
    main()
    pass
