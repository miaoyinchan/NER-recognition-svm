import csv
import json
import time
from contextlib import contextmanager


def load_sentences_from_csv(filepath):
    sentences = []

    with open(filepath, 'r') as fd:
        reader = csv.reader(fd, delimiter=';')
        sentence = []
        for row in reader:
            # There is an empty row after each dot, i.e. sentence end.
            # Then we'll append the sentence to sentences and clear
            # sentence.
            if not row or not all(row):
                sentences.append(sentence)
                sentence = []
                continue
            if len(row) == 1:
                row = (row[0], 'O')
            sentence.append(tuple(row))

    return sentences


def load_gazetteer(filepath):
    with open(filepath, 'r') as fd:
        return json.load(fd)


@contextmanager
def timer(name):
    s = time.monotonic()
    try:
        yield
    finally:
        duration = time.monotonic() - s
    print(f'{name}: {duration:.4f}s')
