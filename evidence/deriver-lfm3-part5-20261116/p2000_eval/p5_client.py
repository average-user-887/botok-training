import json,sys,time,urllib.request,os
URL='http://127.0.0.1:18999'; D=os.path.expanduser('~/p5-eval-20261116')
arms=['base','p4-pct20','p5-final','p5-pct20','p5-pct40','p5-pct60','p5-pct80','p5-pct10','p5-pct30','p5-pct50','p5-pct70','p5-pct90']
files={'p4-pct20':'p4-pct20-f32.gguf','p5-final':'p5-final-f32.gguf',**{f'p5-pct{p}':f'p5-pct{p}-f32.gguf' for p in range(10,100,10)}}
order=[a for a in arms if a!='base']  # lora ids = position in this list
def post(path,body):
    r=urllib.request.Request(URL+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(r,timeout=3600))
prom=json.load(open(D+'/dev_prompts.json'))
os.makedirs(D+'/out',exist_ok=True)
for arm in arms:
    f=D+f'/out/{arm}.json'
    if os.path.exists(f): continue
    lora=[{'id':i,'scale':1.0 if a==arm else 0.0} for i,a in enumerate(order)]
    res=[];t0=time.time()
    for p in prom:
        t=time.time()
        o=post('/completion',{'prompt':p['prompt_ids'],'temperature':0.0,'n_predict':2048,'cache_prompt':False,'lora':lora,'seed':0})
        res.append({'i':p['i'],'text':o['content'],'stop_type':o.get('stop_type'),'stopped_eos':o.get('stopped_eos'),'stopped_limit':o.get('stopped_limit'),'tokens_predicted':o.get('tokens_predicted'),'tokens_evaluated':o.get('tokens_evaluated'),'sec':round(time.time()-t,2)})
        print(arm,p['i'],res[-1]['tokens_predicted'],res[-1]['stop_type'],res[-1]['sec'],flush=True)
    json.dump({'arm':arm,'lora_gguf':files.get(arm),'n_predict':2048,'temperature':0,'rows':res,'sec':round(time.time()-t0,1)},open(f,'w'))
open(D+'/EVAL_DONE','w').write('done')
