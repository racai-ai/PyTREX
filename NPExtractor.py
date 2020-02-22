import re
import sys
import string
import nltk
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from pathlib import Path
from operator import itemgetter

class NPExtractor(object):
    """The class will extract all NPs from a given folder
    with text files (*.txt file extension)."""

    def __init__(self, dname: str, path: str, extract = True, lang = 'en'):
        self._language = lang
        self._npfilename = dname + "-NPs.txt"
        self._wdfilename = dname + "-words.txt"
        self._corpusfilename = dname + "-corpus.txt"
        self._npchunks = {}
        self._wrdfreq = {}

        if Path(self._npfilename).exists():
            self._loadNPs()

        if Path(self._wdfilename).exists():
            self._loadWords()

        if not self._npchunks or not self._wrdfreq:
            self._npchunks = {}
            self._wrdfreq = {}

            # Reads all paragraphs in the training corpus
            # into a list
            self._paragraphs = self._readFolder(path)
            # Preprocess the list and extract the NPs,
            # in a language-dependent manner.
            self._process(self._corpusfilename, lang, extract)
        
    def getNPs(self) -> dict:
        """Return the set of NP chunks found in the corpus
        along with each NP frequency."""
        return self._npchunks

    def getWords(self) -> dict:
        """Return the frequency map of the words in the
        corpus."""
        return self._wrdfreq

    def _loadNPs(self):
        print("Loading NP file {0}...".format(self._npfilename), file = sys.stderr)

        with open(self._npfilename, mode = 'r', encoding = 'utf-8') as f:
            for line in f:
                toks = line.strip().split('\t')
                self._npchunks[toks[0]] = int(toks[1])

    def _loadWords(self):
        print("Loading word frequency file {0}...".format(self._wdfilename), file = sys.stderr)

        with open(self._wdfilename, mode = 'r', encoding = 'utf-8') as f:
            for line in f:
                toks = line.strip().split()
                self._wrdfreq[toks[0]] = int(toks[1])
        # end with

    def dumpWords(self):
        with open(self._wdfilename, mode = 'w', encoding = 'utf-8') as f:
            for (wd, fq) in sorted(self._wrdfreq.items(), key = itemgetter(1), reverse = True):
                f.write("{0}\t{1}\n".format(wd, fq))
        # end with

    def dumpNPs(self):
        with open(self._npfilename, mode = 'w', encoding = 'utf-8') as f:
            for (ck, fq) in sorted(self._npchunks.items(), key = itemgetter(1), reverse = True):
                f.write("{0}\t{1}\n".format(ck, fq))
        # end with

    def _readFolder(self, path: str) -> list:
        """Returns a list of paragraphs from the `path' folder,
        reading all text from the .txt files that are present there."""

        paragraphs = []
        
        for file in list(Path(path).glob('*.txt')):
            para = ''

            with open(file, mode = 'r', encoding = 'utf-8') as f:
                print("  reading file {0}...".format(file), file = sys.stderr)

                while True:
                    line = '#ERR#'

                    while line == '#ERR#':
                        try:
                            line = f.readline()
                        except UnicodeDecodeError:
                            line = '#ERR#'

                    if not line:
                        break

                    line2 = line.strip()
                
                    if line2:
                        para += line
                    elif para:
                        para = para.strip()
                        para = para.replace('“', '"')
                        para = para.replace('”', '"')
                        para = para.replace("’", "'")
                        paragraphs.append(para.strip())
                        para = ''
                # end while with good lines

                if para:
                    para = para.strip()
                    para = para.replace('“', '"')
                    para = para.replace('”', '"')
                    para = para.replace("’", "'")
                    paragraphs.append(para.strip())
            # end with
        # end for
        return paragraphs

    def _process(self, file: str, lang: str, extract: bool):
        """Do the NLP preprocessing of all read paragraphs."""

        lang = lang.lower()

        if lang == 'en' or lang == 'english':
            self._processEnglish(file, extract)

    def _filterNP(self, np: list, prepositions: set) -> list:
        """Removes determiners at the start of the NP
        to make it look like a term for TermEval 2020."""

        npStartTags = {
            "DT", "PDT",
            "WDT", "PRP$"
        }

        npEndTags = {
            "NN", "NNS",
            "NNP", "NNPS"
        }

        # Remove dangling words that are not
        # part of a conventional NP
        while np and np[-1][1] not in npEndTags:
            np.pop()

        # Remove determiners at the left of the NP
        while np and np[0][1] in npStartTags:
            np.pop(0)

        fnp = []

        for wp in np:
            fnp.append(wp[0].lower())

        return fnp

    def _processEnglish(self, file, extract):
        """Do the NLP preprocessing of English text
        performing the NP extraction if 'extract'."""
        
        npStartTags = {
            "DT", "JJ",
            "PDT", "WDT",
            "PRP$"
        }
        npMiddleTags = {
            "JJR", "JJT",
            "RB", "RBR", "RBS",
            "POS", "RP", "VBG",
            "VBN"
        }
        npEndTags = {
            "NN", "NNS",
            "NNP", "NNPS"
        }
        allNPTags = npStartTags.union(npMiddleTags, npEndTags)
        linkPreps = {"of"}
        sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        stop_words = set(stopwords.words('english'))
        word_rx = re.compile("^(?:[a-zA-Z][.]?|[a-zA-Z][a-zA-Z'/.-]*[a-zA-Z.])$")
        
        with open(file, mode = 'w', encoding = 'utf-8') as f:
            for para in self._paragraphs:
                sentences = sentence_splitter.tokenize(para)
            
                for sent in sentences:
                    words = word_tokenize(sent, language = 'english')
                    wordslc = [x.lower() for x in words]
                    tags = pos_tag(words, lang = 'eng')
                    chunk = []
                
                    # Write tokenized sentence into a file for word2vec...
                    f.write("<s> " + " ".join(wordslc) + " </s>\n")

                    if extract:
                        for i in range(len(tags)):
                            w = tags[i][0].lower()
                            t = tags[i][1]

                            if not word_rx.fullmatch(w) and \
                                t in allNPTags and t != 'POS':
                                t = '.'

                            # Fill in the words dictionary
                            if w not in stop_words and \
                                w not in string.punctuation:
                                if not w in self._wrdfreq:
                                    self._wrdfreq[w] = 1
                                else:
                                    self._wrdfreq[w] += 1

                            if (t in npStartTags and chunk) or \
                                (t not in allNPTags and w not in linkPreps and chunk):
                                chkstr = " ".join(self._filterNP(chunk, linkPreps))
                        
                                if chkstr:
                                    if not chkstr in self._npchunks:
                                        self._npchunks[chkstr] = 1
                                    else:
                                        self._npchunks[chkstr] += 1

                                chunk = []
                            elif not chunk:
                                if t in npStartTags or t in npEndTags:
                                    chunk.append(tags[i])
                            else:
                                chunk.append(tags[i])
                        # end for i in current sentence
                        # Add the last chunk as well.
                        if chunk:
                            chkstr = " ".join(self._filterNP(chunk, linkPreps))
                        
                            if chkstr:
                                # Do not add emtpy chunks
                                if not chkstr in self._npchunks:
                                    self._npchunks[chkstr] = 1
                                else:
                                    self._npchunks[chkstr] += 1
                        # end if chunk
                    # end if extract
                # end all sentences
            # end all paragraphs
        # end with
        self._filterChunks(stop_words)

    def _filterChunks(self, stop_words):
        """Will normalize the chunks to remove some
        text preprocessing errors."""
        filtered = {}

        for ck in self._npchunks:
            f = self._npchunks[ck]
            # Replace 's and ' if there are spaces
            ck = ck.replace(" 's", "'s")
            ck = ck.replace(" '", "'")

            if ck not in stop_words and \
                'www.' not in ck and \
                '.com' not in ck and \
                '.org' not in ck and \
                '.edu' not in ck and \
                'http' not in ck:
                filtered[ck] = f
        # end for
        self._npchunks = filtered
