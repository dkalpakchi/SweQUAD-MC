import argparse
import math

import stanza
from stanza.utils.conll import CoNLL
import udon2
from udon2.kernels import ConvPartialTreeKernel

from utils import read_dataset_from_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', default='sv', type=str, help='A language to be used')
    parser.add_argument('-f', '--files', type=str, help='Comma-separated list of JSON files to generate distractors from')
    args = parser.parse_args()

    dep_proc = 'tokenize,lemma,mwt,pos,depparse' if args.lang in ['fi', 'ar'] else 'tokenize,lemma,pos,depparse'
    sv = stanza.Pipeline(lang=args.lang, processors=dep_proc)

    problems = 0
    fname, afname = 'text.conll', 'answer.conll'
    kernel = ConvPartialTreeKernel('GRCT', includeForm=False)

    # lowercasing is fine for individual sentences, but will not work for the whole text, since tokenizer will fail
    for q, a, c, d in read_dataset_from_files(args.files.split(',')):
        c = c.replace(a['context'], '') # replace a paragraph where the correct answer was found
        text = sv(c)
        if not a['text']:
            problems += 1
            continue
        answer = sv(a['text'])

        with open(fname, 'w') as f:
            conll_list = CoNLL.convert_dict(text.to_dict())
            f.write(CoNLL.conll_as_string(conll_list))
        nodes = udon2.ConllReader.read_file(fname)

        with open(afname, 'w') as f:
            conll_list = CoNLL.convert_dict(answer.to_dict())
            f.write(CoNLL.conll_as_string(conll_list))
        a_root = udon2.ConllReader.read_file(afname)[0]
        aa = a_root.children[0]
        N = aa.subtree_size() + 1

        ranks = []
        for n in nodes:
            survivors = n.select_by('upos', aa.upos)
            for s in survivors:
                if s.has_all("feats", str(aa.feats)) and not s.parent.is_root():
                    Ns = s.subtree_size() + 1
                    if (N > 1 and Ns <= 1) or (N <= 1 and Ns > 1): continue
                    aa_k = math.sqrt(kernel(aa, aa))
                    ss_k = math.sqrt(kernel(s, s))
                    ranks.append((s, kernel(aa, s) / (aa_k * ss_k)))
        ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
        # [CLS] and [SEP] is added to emulate BERT's generation for re-using the same evaluation script
        ctx = c.replace('\r\n', ' ').replace('\n', ' ')
        a_text = a['text'].strip().replace('\r\n', ' ').replace('\n', ' ')
        print("[CLS] {} [SEP] {} [SEP] {}".format(ctx, q.strip(), a_text))
        print([r[0].get_subtree_text() for r in ranks[:3]])
        print(d)
        print()