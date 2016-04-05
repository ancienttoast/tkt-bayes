from enum import Enum
from collections import defaultdict
from math import log

#from textblob import TextBlob
#from textblob import Word



class Category(Enum):
  positive = 0
  negative = 1


def words (text):
  """
  punctuation = ",;:.?!"
  neg = ["not", "n't", "no"]

  negation = False
  zen = TextBlob (text)

  for word in zen.tokens:
    w = Word (word).lemmatize().lower()
    if w not in punctuation:
      yield "not_" + w if negation else w

    if w in neg:
      negation = not negation

    if w in punctuation:
      negation = False
  """
  for word in text.split (' '):
    w = word.strip().strip (',.!;?').lower()
    if w:
      yield w


class NaiveBayes:
  def reset (self):
    self.db = defaultdict (lambda: defaultdict (lambda: 0))
    self.total = defaultdict (lambda: 0)

  def __init__ (self):
    self.reset()


  def train (self, text, category):
    for w in words (text):
      self.total[category] += 1
      self.db[w][category] += 1

  def classify (self, text):
    stats = defaultdict (lambda: 0)
    for w in words (text):
      for c in Category:
        stats[c] += log ((self.db[w][c] + 1.0) / (2.0 * self.total[c]))

    return max (stats.keys(), key = lambda x: stats[x])


  def dump (self, file):
    with open (file, 'w') as f:
      for word, stats in self.db.items():
        f.write (word)
        for c in Category:
          f.write (' ')
          f.write (str(stats[c]))
        f.write ('\n')
      f.flush()

  def load (self, file):
    self.reset()
    with open (file, 'r') as f:
      for line in f:
        parts = line.rstrip().rsplit (' ', 2)
        word = parts[0]
        for i, num in enumerate (parts[1:]):
          c = Category(i)
          self.total[c] += int(num)
          self.db[word][c] = int(num)



def train (trainee, file):
  with open (file, "r") as f:
    print ('training', end = "", flush = True)
    i = 1
    for line in f:
      print ('\rtraining', str(i), end = "", flush = True)
      parts = line.split (",", 1)
      trainee.train (parts[1], Category[parts[0]])
      i += 1
    print ('')


def test (bayes, file):
  with open (file, "r") as f:
    print ('testing', end = "", flush = True)
    i = 1
    total = 0
    hit = 0
    for line in f:
      print ('\rtesting', str(i), end = "", flush = True)
      parts = line.split (",", 1)
      r = bt.classify (parts[1])

      hit += 1 if r == Category[parts[0]] else 0
      total += 1
      i += 1
    print ('')

    print ('')
    print (str(hit) + '/' + str(total), "%.2f" % (hit / total * 100) + '%')





bt = NaiveBayes()
train (bt, "train.txt")
bt.dump ("db.db")
#bt.load ("db.db")
test (bt, "test.txt")