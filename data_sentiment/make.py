from __future__ import with_statement
from collections import defaultdict
import sys,os,glob,urllib,re
import cgi,unicodedata

import util
import codecs 
myopen = lambda f,mode='rb',**kwargs: codecs.open(f,mode,encoding='utf8',errors='xmlcharrefreplace',**kwargs)

# /d/panglee/glmnet % for x in $(head -500 ../html/neg);{ cat ../txt_sentoken/neg/*_$x.txt | perl -pe 's/^\s+//; s/\s+$//; s/\s+/\n/g' | awk '{c[$1]+=1} END{for (w in c){ printf("%s",w":"c[w]" ")} print ""  }';} > neg_dev

def uniq_c(seq):
  ret = defaultdict(lambda:0)
  for x in seq:
    ret[x] += 1
  return dict(ret)

word_df = defaultdict(int)
vocab = {}
docids= {}

files = {'pos':myopen("pos.num",'w'), 'neg':myopen("neg.num",'w')}

for tag in ('pos','neg'):
  doc_ids = myopen("%s_doc_ids" % tag).read().split()  #[:100]
  for d in util.counter(doc_ids):
    text = myopen(glob.glob("../txt_sentoken/%s/*_%s.txt" % (tag,d))[0]).read()
    text = re.sub(r'\s+',' ',text.strip())
    words = text.encode('unicode_escape','replace').replace(":","_COLON_").split()
    if d not in docids:
      docids[d] = len(docids)+1
    for w,c in uniq_c(words).items():
      word_df[w] += 1
      if w not in vocab:
        vocab[w] = len(vocab)+1
      print>>files[tag], docids[d], vocab[w], c

with myopen("vocab.txt",'w') as f:
  for w in sorted(vocab, key=lambda w: vocab[w]):
    print>>f, w
with myopen("word_stats.txt",'w') as f:
  for w in sorted(vocab, key=lambda w: vocab[w]):
    print>>f, w, word_df[w]


