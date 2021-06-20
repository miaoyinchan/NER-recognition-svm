import json
import re

from shared import load_sentences_from_csv


def main():
    sentences = load_sentences_from_csv("gro-ner-train.csv") + \
        load_sentences_from_csv("gro-ner-test.csv")

    named_entities = {}
    exclude = ["de", "van", "mannenkoor", "deo", "des", "volk", "duvel",
        "heer", "god", "der", "te", "ten", "'t", "mij", "'n", "stad",
        "zuid", "west", "t", "oost", "den", "recht", "moar", "n'", "d'",
        "hooge", "en", "dokter", "zuud", "post", "veld", "zeun", "in",
        "ter", "het", "brug", "aardbeving", "voor"]

    for sentence in sentences:
        for token, tag in sentence:
            if tag != "O":
                token = token.lower()

                if token.isdigit():
                    continue
                if token in exclude:
                    continue
                if len(token) == 1:
                    continue
                if re.match(r"(?:[a-z]+\.)+", token):
                    continue

                if tag not in named_entities:
                    named_entities[tag] = []
                if token not in named_entities[tag]:
                    named_entities[tag].append(token)

    with open("gazetteer.json", "w") as fd:
        json.dump(named_entities, fd, indent=4)


if __name__ == "__main__":
    main()

