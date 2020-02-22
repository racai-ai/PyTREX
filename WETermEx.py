import re
import sys
import numpy as np
import math
from operator import itemgetter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

class WETermEx(object):
    """Performs term extraction using word embeddings.
    Algorithm:
    1. Train a word embeddings set using the .txt files from the domain
    (tokenized and lower-cased);
    2. Use the English vectors trained with FastText from here:
    https://fasttext.cc/docs/en/english-vectors.html;
    3. Do cosine between any word in the first set and any word
    in the second set and order the list from the most distant vectors
    to the most similar ones."""

    # Word embeddings vectors have the same dimension!
    def __init__(self, general: str, domain: str):
        """Takes the general trained WE file, the word2vec, domain
        specific WE file and a list of words."""

        self._stopwords = set(stopwords.words('english'))
        self._terms = {}
        self._readDomainWordEmbeddings(domain)
        self._readGeneralWordEmbeddings(general)

    def computeTerms(self, npset: dict, tfile: str = None):
        if tfile is not None and Path(tfile).exists():
            self._terms = {}

            with open(tfile, mode = 'r', encoding = 'utf-8') as f:
                for line in f:
                    (t, s) = line.strip().split("\t")
                    self._terms[t] = float(s)
                # end for
            # end while

            return self._terms
        
        # Compute WE distances between single words
        # from the domain and from the general corpus 'bala(nced)'
        termwords = {}

        for word in self._domwe:
            x = self._domwe[word]

            if word in self._genwe:
                y = self._genwe[word]
                # Cosine similarity
                d = 1.0 - (np.dot(x, y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))))
                termwords[word] = d / 2.0
            else:
                termwords[word] = 1.0
        
        # Score each NP with the score of
        # the individual term words
        terms = {}
        indexedterms = {}

        for ck in npset:
            ckwords = ck.split()
            ckscore = 0.0

            for w in ckwords:
                if w in termwords:
                    ckscore += termwords[w]

            terms[ck] = ckscore / float(len(ckwords))

            for w in ckwords:
                if w not in indexedterms:
                    indexedterms[w] = set([ck])
                else:
                    indexedterms[w].add(ck)

        terms2 = {}
        count = 0

        for n1 in terms:
            np1words = n1.split()
            n2set = set()

            for w in np1words:
                if n2set:
                    n2set = n2set.intersection(indexedterms[w])
                else:
                    n2set = n2set.union(indexedterms[w])
            # end build set of all NPs in which words of n1 appear

            for n2 in n2set:
                if len(n2) > len(n1):
                    if n2.startswith(n1 + " ") or \
                        n2.endswith(" " + n1) or \
                        " " + n1 + " " in n2:
                            if not n1 in terms2:
                                terms2[n1] = [terms[n2]]
                            else:
                                terms2[n1].append(terms[n2])
            # end for n2
        # end for n1

        self._terms = {}

        for ck in terms2:
            # Number of words in the NP
            nw = len(ck.split())
            # Mean cosine similarity of all NPs
            # in which ck appeared
            mcs = np.mean(np.array(terms2[ck], dtype = np.float32))
            # Number of NPs in which ck appeared
            nnp = len(terms2[ck])
            # Score of the ck term
            score = nw * mcs * nnp

            self._terms[ck] = score

        # Filtering section
        #self._filterGeneralSingleWordTerms()
        #self._filterNoSingleTermMWTs()

        if tfile is not None:
            # Save terms if we have a file to save to...
            with open(tfile, mode = 'w', encoding = 'utf-8') as f:
                for (t, s) in sorted(self._terms.items(), key = lambda x: x[1], reverse = True):
                    f.write(t + "\t" + str(s) + "\n")
                # end for
            # end while

        return self._terms

    def _filterNoSingleTermMWTs(self):
        """Remove all single-word terms that have a score
        that is not sustained by multi-word terms in which
        the word appears."""
        singlewordterms = set()

        # Collecting the single-word terms first
        for ck in self._terms:
            if " " not in ck:
                singlewordterms.add(ck)

        singlewordindex = {}

        # Build single word to NPs in which the word appears index
        for ck in self._terms:
            ckwords = ck.split()

            if len(ckwords) > 1:
                for w in ckwords:
                    if w in singlewordterms:
                        if w not in singlewordindex:
                            singlewordindex[w] = set([ck])
                        else:
                            singlewordindex[w].add(ck)

        removeneighbours = set()

        for w in singlewordindex:
            neighbours = {}

            for ck in singlewordindex[w]:
                ckwords = ck.split()

                for ckw in ckwords:
                    if ckw != w and ckw != 'of' and \
                        ckw != 'and' and ckw not in neighbours:

                        if ckw not in self._terms:
                            removeneighbours.add(ckw)
            # end for ck
        # end for w
        
        removethem = set()

        for ck in self._terms:
            ckwords = ck.split()

            if len(ckwords) > 1:
                notermwordcount = 0
                ckwordcount = 0

                for w in ckwords:
                    if w in removeneighbours:
                        notermwordcount += 1

                    if w != 'of' and w != 'and':
                        ckwordcount += 1

                rsc = float(notermwordcount) / float(ckwordcount)

                if rsc > 0.5:
                    removethem.add(ck)

        for ck in removethem:
            del self._terms[ck]

    def _filterGeneralSingleWordTerms(self):
        """Remove all single-word terms that only appear
        as heads of other NPs. This means that the word
        is too general."""

        singlewordterms = set()

        # Collecting the single-word terms first
        for ck in self._terms:
            if " " not in ck:
                singlewordterms.add(ck)

        singlewordindex = {}

        # Build single word to NPs in which the word appears index
        for ck in self._terms:
            ckwords = ck.split()

            if len(ckwords) > 1:
                for w in ckwords:
                    if w in singlewordterms:
                        if w not in singlewordindex:
                            singlewordindex[w] = set([ck])
                        else:
                            singlewordindex[w].add(ck)

        removethem = set()

        for w in singlewordindex:
            isgeneral = 0

            for ck in singlewordindex[w]:
                if ck.endswith(" " + w) or \
                    ck.startswith(w + " of ") or \
                    ck.startswith(w + " and ") or \
                    " " + w + " of " in ck or \
                    " " + w + " and " in ck:
                    isgeneral += 1
            # end for ck

            rsc = float(isgeneral) / float(len(singlewordindex[w]))
            
            if rsc >= 0.6:
                removethem.add(w)
        # end for w

        keepthem = singlewordterms.difference(removethem)
        lemmatizer = WordNetLemmatizer()
        savethem = set()

        # Do not remove inflected variants if the lemma is
        # to be kept...
        for wd in removethem:
            lm = lemmatizer.lemmatize(wd)

            if lm in keepthem:
                savethem.add(wd)
        
        for w in removethem:
            if w not in savethem:
                del self._terms[w]

    def _readGeneralWordEmbeddings(self, general):
        print("Loading general word embeddings file {0}...".format(general), file = sys.stderr)

        self._genwe = {}

        with open(general, mode = 'r', encoding = 'utf-8') as f:
            # Skip header
            line = f.readline()
            counter = 0

            for line in f:
                counter += 1
                toks = line.strip().split()
                word = toks.pop(0)

                # Add word only if it was found in the given domain.
                if word in self._domwe:
                    self._genwe[word] = np.array(toks, dtype = np.float32)
                elif word.lower() in self._domwe:
                    self._genwe[word.lower()] = np.array(toks, dtype = np.float32)

                if counter % 100000 == 0:
                    print("  loaded {0} lines".format(counter), file = sys.stderr)
            # end for
        # end with

    def _readDomainWordEmbeddings(self, domain):
        print("Loading domain word embeddings file {0}...".format(domain), file = sys.stderr)

        self._domwe = {}
        word_rx = re.compile("^[a-zA-Z](?:\\w|['/-])*[a-zA-Z]$")
        number_rx = re.compile("[0-9]")

        with open(domain, mode = 'r', encoding = 'utf-8') as f:
            # Skip header
            line = f.readline()

            for line in f:
                toks = line.strip().split()
                word = toks.pop(0)

                # Word is already lower-cased here!
                if word_rx.match(word) and \
                    not number_rx.search(word) and \
                    word not in self._stopwords:
                    self._domwe[word] = np.array(toks, dtype = np.float32)
                #else:
                #    print("  rejecting word {0}".format(word), file = sys.stderr)
            # end for
        # end with
