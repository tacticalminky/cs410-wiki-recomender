from crawler_v2 import start_crawler
from link_ranking import calc_link_ranks
from processer import create_vocab, reduce_and_sort

def main() -> None:
    start_crawler()

    inv_idx, vocab = create_vocab()

    reduce_and_sort(inv_idx, vocab)

    del inv_idx, vocab

    calc_link_ranks()

    return

if __name__ == "__main__":
    main()
    pass
