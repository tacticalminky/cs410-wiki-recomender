import nltk
import pandas as pd
import requests
import re
import os

from threading import Thread, Lock, Event
from queue import Queue

from bs4 import BeautifulSoup

from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

NUM_TREADS = 4

MAX_NUM_ITERATIONS = int(1e3)

base_url = 'https://en.wikipedia.org'

seed_links = (
    'https://en.wikipedia.org/wiki/University_of_Illinois_Urbana-Champaign',
    'https://en.wikipedia.org/wiki/Computer_science'
)

output_file = './data/output.csv'

def thread_task(queue: Queue[str], finished: Event, lock: Lock, visited: set[str], pbar: tqdm) -> None:
    urls:    list[str] = []
    titles:  list[str] = []
    content: list[str] = []

    while not finished.is_set():
        # TODO: check
        with lock:
            if finished.is_set() or len(visited) >= MAX_NUM_ITERATIONS:
                finished.set()
                break

            link = queue.get()
            while link in visited:
                link = queue.get()

            visited.add(link)

        urls.append(link)

        html = requests.get(link).text
        page = BeautifulSoup(html, 'html.parser')

        titles.append(page.find(id='firstHeading').text)

        body = page.find(id='bodyContent')

        found_links = set()
        for sub_link in body.find_all('a'):
            if not sub_link.has_attr('href'): continue

            sub_link = sub_link['href']

            if (sub_link.startswith('/wiki/File:')
                or sub_link.startswith('/wiki/Help:')
                or sub_link.startswith('/wiki/Special:')
                or sub_link.startswith('/wiki/Template:')
                or sub_link.startswith('/wiki/Template_talk:')):
                continue

            if sub_link.startswith('/wiki/'):
                sub_link = base_url + sub_link
                if sub_link not in visited and sub_link not in found_links:
                    found_links.add(sub_link)

        for sub_link in found_links:
            queue.put(sub_link)

        text = ''
        for par in body.find_all('p'):
            text += ' ' + par.text

        # remove new line chars, non-alphanumeric chars, and nums
        text = re.sub(r'\n|\r', ' ', text)
        text = re.sub(r"'[\w]*|[^\w\s]+", ' ', text)
        text = re.sub(r'\d+[\w]*', ' ', text)

        # remove extra whitespace and strip
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        text = text.lower()

        # tokenize and remove stopwords
        word_tokens = word_tokenize(text)
        filtered = [ w for w in word_tokens if not w in stop_words ]
        text = ' '.join(filtered)

        content.append(text)

        with lock:
            pbar.update(1)

        # output every
        if len(urls) >= 25:
            data = pd.DataFrame({'url': urls, 'title': titles, 'content': content})
            data.to_csv(output_file, mode='a', index=False, header=False)

            urls.clear()
            titles.clear()
            content.clear()

    # output contents
    if len(urls) > 0:
        data = pd.DataFrame({'url': urls, 'title': titles, 'content': content})
        data.to_csv(output_file, mode='a', index=False, header=False)

    return

def main() -> None:
    if os.path.exists(output_file):
        os.remove(output_file)

    queue = Queue()
    for link in seed_links:
        queue.put(link)

    finished = Event()
    lock     = Lock()
    visited  = set()

    pbar = tqdm(total=MAX_NUM_ITERATIONS)

    threads: list[Thread] = []
    for _ in range(NUM_TREADS):
        thread = Thread(target=thread_task, args=(queue, finished, lock, visited, pbar))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return

if __name__ == "__main__":
    main()
    pass
