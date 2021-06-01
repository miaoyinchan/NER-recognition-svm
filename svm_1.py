#!/usr/bin/env python

import csv
import re
import time
from contextlib import contextmanager

import numpy as np
from nltk import ngrams
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

import gazetteer


def main():
    with timer('load data'):
        train_sentences = load_sentences_from_csv('gro-ner-train.csv')
        train_labels = extract_labels(train_sentences)
        test_sentences = load_sentences_from_csv('gro-ner-test.csv')
        test_labels = extract_labels(test_sentences)
        train_y = np.array(train_labels)
        test_y = np.array(test_labels)

    with timer('extract features'):
        train_x = extract_features(train_sentences)
        test_x = extract_features(test_sentences)

    del train_sentences
    del test_sentences

    with timer('setup pipeline'):
        pipeline = Pipeline([
            ('dvect', DictVectorizer(sparse=True)),
            # ('tfidf', TfidfTransformer(smooth_idf=True)),
            ('clf', LinearSVC(max_iter=100000, class_weight='balanced')),
        ])

    with timer('pipeline.fit'):
        pipeline.fit(train_x, train_y)

    with timer('pipeline.predict'):
        predictions = pipeline.predict(test_x)

    print('Set 1')
    print(classification_report(test_y, predictions, zero_division=0))

    # ps = {
    #     'clf__C': [1, 5, 10, 20, 25, 30, 35, 100],
    # }
    # gs = GridSearchCV(pipeline, param_grid=ps, n_jobs=2, verbose=3, cv=10)

    # with timer('gs.fit'):
    #     gs.fit(train_x, train_y)
    #     print(f'Best score: {gs.best_score_}')
    #     print(f'Best C: {gs.best_estimator_["clf"].C}')

    # with timer('gs.predict'):
    #     predictions = gs.predict(test_x)

    # print(classification_report(test_y, predictions, zero_division=0))


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


def extract_labels(sentences):
    return [
        label
        for sentence in sentences
        for _, label in sentence
    ]


def extract_features(sentences):
    results = []

    for sentence in sentences:
        results.extend(extract_features_from_sentence(sentence))

    return results


def extract_features_from_sentence(sentence):
    results = []
    offsets = (-2, -1, 0, 1, 2)

    # Adding padding to sentence list so the current token is always
    # in the middle of the ngram tuple.
    ngram_len = 5
    padding = [None, None]
    token_ngrams = ngrams(padding + sentence + padding, ngram_len)

    correction = len(offsets) // 2

    for i, tokens in enumerate(token_ngrams):
        features = {}

        for offset in offsets:
            token_and_label = tokens[offset + correction]
            if token_and_label:
                token = token_and_label
                if len(token) == 2:
                    token, _ = token

                features[f'{offset}:token.text'] = token
                features[f'{offset}:token.prefix_4'] = feature_prefix_4(token)
                features[f'{offset}:token.suffix_4'] = feature_token_suffix_4(token)
                features[f'{offset}:token.is_short'] = feature_is_short(token)
                features[f'{offset}:token.is_number'] = feature_is_number(token)
                features[f'{offset}:token.is_punctuation'] = feature_is_punctuation(token)
                features[f'{offset}:token.first_is_uppercase'] = feature_first_is_uppercase(token)
                features[f'{offset}:token.is_uppercase'] = feature_is_uppercase(token)
                features[f'{offset}:token.is_lowercase'] = feature_is_lowercase(token)
                features[f'{offset}:token.is_abbreviation_or_initial'] = feature_is_abbreviation_or_initial(token)

                if offset == 0:
                    result = feature_start_of_sentence(token)
                    if result:
                        features[f'{offset}:token.start_of_sentence'] = result
                    result = feature_is_possible_first_name(token)
                    if result:
                        features[f'{offset}:token.is_possible_first_name'] = result
                    result = feature_is_possible_last_name(token)
                    if result:
                        features[f'{offset}:token.is_possible_last_name'] = result
                    result = feature_is_possible_first_location_name(token)
                    if result:
                        features[f'{offset}:token.is_possible_first_location_name'] = result
                    result = feature_is_possible_last_location_name(token)
                    if result:
                        features[f'{offset}:token.is_possible_last_location_name'] = result
                    result = feature_is_possible_first_org_name(token)
                    if result:
                        features[f'{offset}:token.is_possible_first_org_name'] = result
                    result = feature_is_possible_last_org_name(token)
                    if result:
                        features[f'{offset}:token.is_possible_last_org_name'] = result
                    result = feature_is_possible_first_misc_name(token)
                    if result:
                        features[f'{offset}:token.is_possible_first_misc_name'] = result
                    result = feature_is_possible_last_misc_name(token)
                    if result:
                        features[f'{offset}:token.is_possible_last_misc_name'] = result

        results.append(features)

    return results


def feature_prefix_3(token):
    return token.lower()[:3]


def feature_token_suffix_3(token):
    return token.lower()[-3:]


def feature_prefix_4(token):
    return token.lower()[:4]


def feature_token_suffix_4(token):
    return token.lower()[-4:]


def feature_start_of_sentence(position):
    return position == 0


def feature_end_of_sentence(position, sentence):
    return position == (len(sentence) - 2)


def feature_is_short(token):
    return len(token) < 4


def feature_is_number(token):
    return re.match(r'\d+(?:\.\d+)?', token) is not None


def feature_is_punctuation(token):
    return token in '.,?!-:;\'"“”‘’'


def feature_first_is_uppercase(token):
    # Also account for d'ChristenUnie.
    return token[0].isupper() or re.match(r'[a-z][\'’][A-Z].+', token) is not None


def feature_is_uppercase(token):
    return token.isupper()


def feature_is_lowercase(token):
    return token.islower()


def feature_is_abbreviation_or_initial(token):
    return re.match(r'(?:\w+\.)+', token) is not None


def feature_vowel_ratio(token):
    # consonants = 'bcdfghjklmnpqrstvwxyz'
    vowels = 'aeiou'

    count = sum([c in vowels for c in token])
    ratio = count / len(token)

    return ratio


def feature_is_possible_first_name(token):
    return token.lower() in gazetteer.b_per


def feature_is_possible_last_name(token):
    return token.lower() in gazetteer.i_per


def feature_is_possible_first_location_name(token):
    return token.lower() in gazetteer.b_loc


def feature_is_possible_last_location_name(token):
    return token.lower() in gazetteer.i_loc


def feature_is_possible_first_org_name(token):
    return token.lower() in gazetteer.b_org


def feature_is_possible_last_org_name(token):
    return token.lower() in gazetteer.i_org


def feature_is_possible_first_misc_name(token):
    return token.lower() in gazetteer.b_misc


def feature_is_possible_last_misc_name(token):
    return token.lower() in gazetteer.i_misc


@contextmanager
def timer(name):
    s = time.monotonic()
    try:
        yield
    finally:
        duration = time.monotonic() - s
    print(f'{name}: {duration:.4f}s')


if __name__ == "__main__":
    with timer('main'):
        main()
