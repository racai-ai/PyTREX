import sys
from NPExtractor import NPExtractor
from WETermEx import WETermEx

# Radu ION: note -> This is the WEMBF algorithm from our TermEval 2020 LREC paper.
# Do not forget to unzip data/data.zip before running any of this!

GOLDPATH = 'data/ACTER_version1_1_only_train/en/{0}/annotations/{0}_en_terms_nes.ann'
DATAPATH = 'data/ACTER_version1_1_only_train/en/{0}/texts/all'
DOMWORDEMB = 'data/ACTER_version1_1_only_train/en/{0}/{0}-word-embeddings.txt'
GENWEMBFILE = DOMWORDEMB.format('bala')

def readAnswers(file: str) -> set:
    answers = set()

    with open(file, mode = 'r', encoding = 'utf-8') as f:
        for line in f:
            answers.add(line.strip())
        # end for
    # end with

    return answers

def evalSystem(system: dict, gold: set, score = 0.0):
    allgood = 0
    good = {}
    syscounts = {}
    gldcounts = {}

    for t in gold:
        nw = len(t.split())

        if nw not in gldcounts:
            gldcounts[nw] = 1
        else:
            gldcounts[nw] += 1

    for t in system:
        nw = len(t.split())

        if system[t] >= score:
            if nw not in syscounts:
                syscounts[nw] = 1
            else:
                syscounts[nw] += 1

            if t in gold:
                allgood += 1

                if nw not in good:
                    good[nw] = 1
                else:
                    good[nw] += 1

    prec = float(allgood) / float(len(system))
    rec = float(allgood) / float(len(gold))
    fone = 0.0

    if prec + rec > 0.0:
        fone = 2 * prec * rec / (prec + rec)

    print(file = sys.stderr)
    print("Global P = {0:.2f}%".format(prec * 100.0), file = sys.stderr)
    print("Global R = {0:.2f}%".format(rec * 100.0), file = sys.stderr)
    print("Global F1 = {0:.2f}%".format(fone * 100.0), file = sys.stderr)
    print(file = sys.stderr)

    for (nw, h) in sorted(good.items(), key = lambda x: x[0]):
        pr = float(h) / float(syscounts[nw])
        rc = float(h) / float(gldcounts[nw])
        f1 = 0.0

        if pr + rc > 0.0:
            f1 = 2 * pr * rc / (pr + rc)

        print("P for {0} word terms = {1:.2f}%".format(nw, pr * 100.0), file = sys.stderr)
        print("R for {0} word terms = {1:.2f}%".format(nw, rc * 100.0), file = sys.stderr)
        print("F1 for {0} word terms = {1:.2f}%".format(nw, f1 * 100.0), file = sys.stderr)
        print(file = sys.stderr)


#for dom in ['bala']:
#for dom in ['corp', 'equi', 'wind', 'hf']:
#    print("Extracting NPs from domain {0}...".format(dom), file = sys.stderr)
#    path = DATAPATH.format(dom)
#    npx = NPExtractor(dom, path)
#    npx.dumpWords()
#    npx.dumpNPs()

dname = 'hf'
npx = NPExtractor(dname, DATAPATH.format(dname))
wetx = WETermEx(GENWEMBFILE, DOMWORDEMB.format(dname))
t = wetx.computeTerms(npx.getNPs(), dname + "-terms.txt")
#g = readAnswers(GOLDPATH.format(dname))
#evalSystem(t, g, 0.1)
