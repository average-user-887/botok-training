import json,sys,re
def toks(s): return len(s)//4
ok=True; n=0
CL={"short":(500,1000,3,20,3,40),"medium":(1500,3000,8,36,8,55),"long":(3400,5500,30,70,30,110)}
for i,l in enumerate(open(sys.argv[1]),1):
    if not l.strip(): continue
    try: x=json.loads(l)
    except Exception as e: print(i,"bad json");ok=False;continue
    n+=1; e=[]
    for k in("id","peer","messages","gold"):
        if k not in x: e.append("missing "+k)
    if e: print(i,e);ok=False;continue
    m=x["messages"]; ts=[a["ts"] for a in m]
    if ts!=sorted(ts) or len(set(ts))!=len(ts): e.append("ts not strictly increasing")
    if any(not re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d",t) for t in ts): e.append("ts format")
    spk={a["peer"] for a in m}
    if x["gold"] and x["peer"] not in spk: e.append("target silent but gold nonempty")
    mt=sum(toks(a["content"]) for a in m); cls=x["id"].split("-")[1] if x["id"].count("-")>=2 else None
    if cls in CL:
        a,b,c,d,g1,g2=CL[cls]
        if not(a<=mt<=b*1.25): e.append(f"{cls} msg tokens {mt} out of range")
        if not(c<=len(m)<=d+6): e.append(f"{cls} n_msgs {len(m)}")
        if len(x["gold"])>g2*1.2: e.append("too many gold")
        if x["gold"] and len(x["gold"])<g1*0.6 : e.append(f"too few gold {len(x['gold'])}")
    low=" ".join(a["content"].lower() for a in m)
    for g in x["gold"]:
        if not g.lower().startswith(x["peer"].lower()) and x["peer"].lower() not in g.lower(): e.append("gold lacks peer id: "+g[:40]); break
    if len(set(g.lower() for g in x["gold"]))!=len(x["gold"]): e.append("duplicate gold")
    ug=0
    for g in x["gold"]:
        w=[t for t in re.findall(r"[a-z0-9]{4,}",g.lower()) if t!=x["peer"].lower()]
        if w and sum(t in low for t in w)/len(w)<0.35: ug+=1
    if ug>max(2,0.15*len(x["gold"])): e.append(f"{ug} gold weakly grounded in messages")
    if e: print(i,x["id"],e);ok=False
print("OK" if ok else "FIX", n)
sys.exit(0 if ok else 1)
