import json,glob,os,importlib.util,sys
spec=importlib.util.spec_from_file_location('td','train_deriver_490c.py  # = evidence-deriver-run-20261115 code/train_deriver.py @490c643'); td=importlib.util.module_from_spec(spec); sys.argv=['x']; spec.loader.exec_module(td)
rows=[json.loads(l) for l in open('x/dev.sft.jsonl')]
order=['base','p4-pct20']+[f'p5-pct{p}' for p in range(10,100,10)]+['p5-final']
print('arm valid tp pn gn P R F1 limit_hits')
for a in order:
    d=json.load(open(f'out/{a}.json'));v=tp=pn=gn=gtp=lim=0
    for r in d['rows']:
        fin=(r['stop_type']=='eos'); lim+=int(not fin)
        ok,t,p,g,gt,_,_=td.score_prediction('deriver',rows[r['i']],r['text'].strip(),finished=fin)
        v+=int(ok);tp+=t;pn+=p;gn+=g;gtp+=gt
    P=tp/pn if pn else 0;R=gtp/gn if gn else 0;F=2*P*R/(P+R) if P+R else 0
    print(f'{a:10s} {v}/24 {tp} {pn} {gn} {P:.3f} {R:.3f} {F:.3f} {lim}')
