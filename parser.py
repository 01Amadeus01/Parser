import nltk
import sys
from nltk.tokenize import word_tokenize

nltk.download('punkt')

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
AdjP -> Adj | Adj AdjP | Adj N | Det Adj N | Det AdjP
AdvP -> V Adv | Adv AdjP | Adv V
PP -> P NP | P Det AdjP | P Det AdvP | P AdjP | P AdvP | PP PP
S -> S Conj S
NP -> N | Det N
VP -> V | VP NP
S -> NP VP NP PP
S -> NP VP NP
S -> NP VP PP Conj N
S -> NP VP AdjP
S -> NP VP PP | N V PP PP
S -> NP AdvP NP
S -> NP VP PP Adv
S -> NP Adv VP Conj NP VP Adv
S -> NP AdvP Conj VP NP
S -> NP VP AdjP PP Conj VP NP PP
S -> NP VP AdjP PP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    sentence = sentence.lower()
    sentence = word_tokenize(sentence)
    sentence = [word for word in sentence if any(char.isalpha() for char in word)]
    return sentence


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    c = 0
    np = []
    for s in tree.subtrees():
        if s.label() == 'NP':
            for v in s.subtrees():
                if v.label() == 'NP' and v != s:
                    c = 1
            if c == 0:
                np.append(s)
        c = 0
    return np


if __name__ == "__main__":
    main()
