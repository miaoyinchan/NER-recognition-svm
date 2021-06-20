import re

import click
import numpy as np
from nltk import ngrams
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.class_weight import compute_class_weight

from shared import load_sentences_from_csv, load_gazetteer, timer


GAZETTEER = load_gazetteer('gazetteer.json')


@click.command()
@click.option('--enable-cv/--disable-cv', default=True)
@click.option('--enable-gazetteer/--disable-gazetteer', default=True)
def main(enable_cv, enable_gazetteer):
    if enable_cv:
        print('Cross-validation enabled!')
    if enable_gazetteer:
        print('Gazetter enabled!')

    with timer('load data'):
        train_sentences = load_sentences_from_csv('gro-ner-train.csv')
        train_labels = extract_labels(train_sentences)
        test_sentences = load_sentences_from_csv('gro-ner-test.csv')
        test_labels = extract_labels(test_sentences)

    with timer('featurize data'):
        # 100% training data.
        X = extract_features(train_sentences, enable_gazetteer)
        y = np.array(train_labels)

        # 100% test data.
        X_test = extract_features(test_sentences, enable_gazetteer)
        y_test = np.array(test_labels)

        # 90% training data, 10% dev data.
        X_train, X_dev, y_train, y_dev = train_test_split(
            X,
            y,
            train_size=0.1,
            random_state=1,
        )

    with timer('setup training model'):
        model = Pipeline([
            ('dvect', DictVectorizer(sparse=True)),
            ('clf', LinearSVC(
                C=1.0,
                max_iter=1000000,
                class_weight='balanced',
                random_state=1,
                dual=False,
            )),
        ])
        print('training model parameters:')
        for k, v in model.get_params().items():
            print(f'  {k}: {v}')

    with timer('training model.fit'):
        model.fit(X_train, y_train)

    with timer('training model.predict'):
        predictions = model.predict(X_dev)

    with timer('training results'):
        print('Training classification report:')
        print(classification_report(y_dev, predictions, zero_division=0))

    if enable_cv:
        print('Cross-validation')

        param_grid = {
            'clf__C': [0.001, 0.10, 0.1, 10, 25, 50],
        }

        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring='f1_macro',
            n_jobs=2,
            verbose=3,
            cv=5,
        )

        with timer('grid_search.fit'):
            grid_search.fit(X_train, y_train)
            print(f'Best cross-validation score: {grid_search.best_score_}')
            print('Tuned parameters:')
            for k, v in grid_search.best_params_.items():
                print(f'  {k}: {v}')

        with timer('setup final model'):
            model = Pipeline([
                ('dvect', DictVectorizer(sparse=True)),
                ('clf', LinearSVC(
                    class_weight='balanced',
                    max_iter=1000000,
                    random_state=1,
                    dual=False,
                )),
            ])
            model.set_params(**grid_search.best_params_)
            print('final model parameters:')
            for k, v in model.get_params().items():
                print(f'  {k}: {v}')

        with timer('final model.fit'):
            model.fit(X, y)

        with timer('final model.predict'):
            predictions = model.predict(X_test)

        with timer('final results'):
            print('Training classification report:')
            print(classification_report(y_test, predictions, zero_division=0))


def extract_labels(sentences):
    return [
        label
        for sentence in sentences
        for _, label in sentence
    ]


def extract_features(sentences, enable_gazetteer):
    results = []

    for sentence in sentences:
        results.extend(extract_features_from_sentence(sentence, enable_gazetteer))

    return results


def extract_features_from_sentence(sentence, enable_gazetteer):
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

                # Features for focus token, two previous tokens, and two next tokens.
                features[f'{offset}:token'] = token.lower()
                features[f'{offset}:token.istitle'] = token.istitle()
                features[f'{offset}:token.isupper'] = token.isupper()

                # Features for focus token.
                if offset == 0:
                    features[f'{offset}:token.has_punctuation'] = has_punctuation(token)
                    features[f'{offset}:token.is_abbre'] = is_abbre(token)
                    features[f'{offset}:token.start_of_sentence'] = start_of_sentence(token)

                    if enable_gazetteer:
                        features[f'{offset}:token.is_BPER_gaz'] = is_BPER_gaz(token)
                        features[f'{offset}:token.is_IPER_gaz'] = is_IPER_gaz(token)
                        features[f'{offset}:token.is_BLOC_gaz'] = is_BLOC_gaz(token)
                        features[f'{offset}:token.is_ILOC_gaz'] = is_ILOC_gaz(token)
                        features[f'{offset}:token.is_BORG_gaz'] = is_BORG_gaz(token)
                        features[f'{offset}:token.is_IORG_gaz'] = is_IORG_gaz(token)
                        features[f'{offset}:token.is_BMISC_gaz'] = is_BMISC_gaz(token)
                        features[f'{offset}:token.is_IMISC_gaz'] = is_IMISC_gaz(token)

                    features[f'{offset}:token.is_function_word'] = is_function_word(token)

        results.append(features)

    return results


def start_of_sentence(position):
    return position == 0


def has_punctuation(token):
    for c in '.,?!-:;\'"“”‘’':
        if c in token:
            return True
    return False


def is_abbre(token):
    return re.match(r'(?:\w+\.)+', token) is not None


def is_BPER_gaz(token):
    return token.lower() in GAZETTEER['B-PER']


def is_IPER_gaz(token):
    return token.lower() in GAZETTEER['I-PER']


def is_BLOC_gaz(token):
    return token.lower() in GAZETTEER['B-LOC']


def is_ILOC_gaz(token):
    return token.lower() in GAZETTEER['I-LOC']


def is_BORG_gaz(token):
    return token.lower() in GAZETTEER['B-ORG']


def is_IORG_gaz(token):
    return token.lower() in GAZETTEER['I-ORG']


def is_BMISC_gaz(token):
    return token.lower() in GAZETTEER['B-MISC']


def is_IMISC_gaz(token):
    return token.lower() in GAZETTEER['I-MISC']


def is_function_word(token):
    return token.lower() in ['van', 'de', 'ter', 'der', 'te']


if __name__ == "__main__":
    with timer("main"):
        main()

