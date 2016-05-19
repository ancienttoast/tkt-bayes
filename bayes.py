from enum import Enum
from collections import defaultdict
from math import log
import re
import pickle

import sys
from textblob import TextBlob
from textblob import Word


class Category(Enum):
    positive = 0
    negative = 1


#ez a függvény a bemenetén kapott szövegen végig iterál szavaként, majd visszatér a szavak listájával
#a negatív módosító szópárokkal együtt (pl not good) - ezek egynek számítanak
def words_textblob(text):
    punctuation = ",;:.?!"
    neg = ["not", "n't", "no"]

    negation = False
    blob = TextBlob(text)

    for word in blob.tokens:
        w = Word(word).lemmatize().lower()
        if not re.match("<.*>", w):
            if w not in punctuation:
                yield "not_" + w if negation else w

            if w in neg:
                negation = not negation

            if w in punctuation:
                negation = False

#ez a függvény a bemenetén kapott szöveget szóközök mentén, az írásjeleket figyelembe véve, visszatér a szavak listájával
def words_custom(text):
    for word in text.split(' '):
        w = word.strip().strip(',.!;?()').lower()
        # If there is a word after stripping, and it is not an HTML tag, then yield it.
        if w and not re.match("<.*>", w):
            yield w







class NaiveBayes:
    def __init__(self, word_function=words_textblob):
        self.reset()
        self.words = word_function

    def reset(self):
        self.db = defaultdict(lambda: defaultdict(lambda: 0))
        self.total = defaultdict(lambda: 0)

    def train(self, text, category):
        for w in self.words(text):
            self.total[category] += 1
            self.db[w][category] += 1

    def classify(self, text):
        stats = defaultdict(lambda: 0)
        for w in self.words(text):
            for c in Category:
                stats[c] += log((self.db[w][c] + 1.0) / (2.0 * self.total[c]))

        return max(stats.keys(), key=lambda x: stats[x])

    def dump(self, file):
        with open(file, 'w', encoding="UTF-8") as f:
            for word, stats in self.db.items():
                f.write(word)
                for c in Category:
                    f.write(' ')
                    f.write(str(stats[c]))
                f.write('\n')
            f.flush()

    def load(self, file):
        self.reset()
        with open(file, 'r', encoding="UTF-8") as f:
            for line in f:
                parts = line.rstrip().rsplit(' ', 2)
                word = parts[0]
                for i, num in enumerate(parts[1:]):
                    c = Category(i)
                    self.total[c] += int(num)
                    self.db[word][c] = int(num)

def train(trainee, file):
    with open(file, "r", encoding="UTF-8") as f:
        print('training', end="", flush=True)
        i = 1
        for line in f:
            print('\rtraining', str(i), end="", flush=True)
            parts = line.split(",", 1)
            trainee.train(parts[1], Category[parts[0]])
            i += 1
        print('')


def test(bayes, file, output=sys.stdout):
    with open(file, "r", encoding="UTF-8") as f:
        print('testing', end="", flush=True)
        i = 1
        total = 0
        hit = 0
        for line in f:
            print('\rtesting', str(i), end="", flush=True)
            parts = line.split(",", 1)
            r = bayes.classify(parts[1])

            hit += 1 if r == Category[parts[0]] else 0
            total += 1
            i += 1
        print('')

        print('')
        output_file = open(output, "w") if type(output) == type("") else sys.stdout
        print(str(hit) + '/' + str(total), "%.2f" % (hit / total * 100) + '%', file=output_file)
        if type(output) == type(""):
            output_file.close()

def main():
    db_name = ""
    bt = NaiveBayes()

    mode = "custom"
    type = "testasa"

    if mode == "custom":
        bt.words = words_custom            # feldarabolja a szöveget egy szólistává
        db_name = "db_custom.db"           # elnevezzük az adatbázist
    else:
        bt.words = words_textblob       # visszatér a szavak listájával, a not meg egyéb negatív szópárt is figyelembe véve
        db_name = "db_textblob.db"      # elnevezzük az adatbázist

    if type == "train":
        train(bt, "train.txt")      #a betanítószöveget odaadjuk a Bayes modellnek
        bt.dump(db_name)
    elif type == "test":
        bt.load(db_name)        #ha már betanulta, akkor jöhet a tesztelés
        test(bt, "test.txt", mode + "_result.txt")      #tesztelés eredménye a result fileban lesz
    else:
    	bt.load(db_name)        #ha már betanulta, akkor jöhet a tesztelés
    	print (bt.classify (sys.argv[1]))


if __name__ == '__main__':
    main()
