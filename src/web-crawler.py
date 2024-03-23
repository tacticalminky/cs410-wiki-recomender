import pandas as pd
import requests
import re
import threading

from bs4 import BeautifulSoup

NUM_TREADS = 4

MAX_NUM_ITERATIONS = int(1e3)

base_url = 'https://en.wikipedia.org/wiki/'

seed_links = (
    'https://en.wikipedia.org/wiki/University_of_Illinois_Urbana-Champaign',
    'https://en.wikipedia.org/wiki/Computer_science'
)

def thread_func() -> None:
    return

def main() -> None:
    # TODO: clear output.csv

    # TODO: parallize

    urls:    list[str] = []
    titles:  list[str] = []
    content: list[str] = []

    for link in seed_links:
        urls.append(link)

        html = requests.get(link).text
        page = BeautifulSoup(html, 'html.parser')

        # TODO: add links to queue

        titles.append(page.find(id='firstHeading').text)

        text = ''
        for par in page.find(id='bodyContent').find_all('p'):
            text += ' ' + par.text

        # remove new line chars, non-alphanumeric chars, and nums
        text = re.sub(r'\n|\r', ' ', text)
        text = re.sub(r"'[\w]*|[^\w\s]+", ' ', text)
        text = re.sub(r'\d+[\w]*', ' ', text)

        # remove extra whitespace and strip
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        text = text.lower()

        content.append(text)

    # output contents
    # TODO: output every ? iterations and clear lists
    data = pd.DataFrame({'url': urls, 'title': titles, 'content': content})
    data.to_csv('./data/output.csv', mode='a', index=False, header=False)

    # TODO: output at end

    return

if __name__ == "__main__":
    main()
    pass
