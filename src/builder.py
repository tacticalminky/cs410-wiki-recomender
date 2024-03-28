from crawler import crawl
from processer import create_vocab, reduce_and_sort

def main() -> None:
    crawl()
    print()

    inv_idx, vocab = create_vocab()
    print()

    reduce_and_sort(inv_idx, vocab)

    # TODO: PageRank

    return

if __name__ == "__main__":
    main()
    pass
