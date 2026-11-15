import json,glob,random,subprocess,sys,os
sys.path.insert(0,'.')
from deriver_prompt import minimal_deriver_prompt
ok=[];bad=[]
for f in sorted(glob.glob('synthetic_v2/*.jsonl')):
    r=subprocess.run([sys.executable,'synthetic_v2/validate.py',f],capture_output=True,text=True)
    (ok if r.stdout.strip().splitlines()[-1:] and r.stdout.strip().splitlines()[-1].startswith('OK') else bad).append(f)
print("bad:",bad)
rows=[]
for f in ok:
    for l in open(f):
        x=json.loads(l)
        msgs="\n".join(f"{m['ts']} {m['peer']}: {m['content']}" for m in x['messages'])
        p=minimal_deriver_prompt(peer_id=x['peer'],messages=msgs)
        rows.append({"id":x['id'],"messages":[{"role":"user","content":p},{"role":"assistant","content":json.dumps({"explicit":[{"content":g} for g in x['gold']]},ensure_ascii=False)}]})
rnd=random.Random(11); rnd.shuffle(rows)
ev=[];tr=[]
cnt={}
for r in rows:
    c=r['id'].split('-')[1]; cnt[c]=cnt.get(c,0)+1
    (ev if cnt[c]%20==0 else tr).append(r)
for f,d in (("sft_train.jsonl",tr),("sft_eval.jsonl",ev)):
    for l in open(f): d.append(json.loads(l))
rnd.shuffle(tr)
for n,d in (("v2_train.jsonl",tr),("v2_dev.jsonl",ev)):
    open(n,'w').write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in d))
emp=sum(1 for r in tr if r['messages'][1]['content']=='{"explicit": []}')
tok=sum(len(r['messages'][0]['content'])+len(r['messages'][1]['content']) for r in tr)//4
print(len(tr),len(ev),"empty",emp,"approx tok",tok)
