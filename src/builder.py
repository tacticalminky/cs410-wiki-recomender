from crawler import crawl
from processer import calc_PageRanks, create_vocab, reduce_and_sort

def main() -> None:
    crawl()

    inv_idx, vocab = create_vocab()

    reduce_and_sort(inv_idx, vocab)

    del inv_idx, vocab

    calc_PageRanks()

    return

if __name__ == "__main__":
    main()
    pass
