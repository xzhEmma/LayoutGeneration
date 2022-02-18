from nltk.corpus import wordnet as wn

p = wn.synsets('sky')
a = wn.synsets('bowl')
c = max([0 if b.path_similarity(q) is None else b.path_similarity(q) for b in a for q in p])
print(c)