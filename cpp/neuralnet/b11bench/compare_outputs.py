"""Compare raw heads and board-valid probabilities without requiring NumPy.

Raw files are direct_bench.cpp dumps: FP32 pass[B,2], policy[B,S,2],
value[B,3], score[B,6], ownership[B,S], in that order (S = len * len).
Optional --corpus is the exact direct_bench input file: headerless FP32 rows of
NHWC spatial[S,22] followed by global[19]. Channel zero is the binary board mask.
--row-offset selects the first matching corpus row; indices wrap like the replay
harness. The current replay uses symmetry zero. Policy probabilities/top1 then
exclude padded cells but always retain pass. Raw-head metrics still cover every
dumped element; ownership_valid additionally covers only real board cells.
"""
import argparse,array,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--batch',type=int,required=True);p.add_argument('--len',type=int,default=19)
p.add_argument('--corpus',help='Matching direct_bench NHWC22 + global19 FP32 input corpus')
p.add_argument('--row-offset',type=int,default=0,help='First corpus row for this raw dump (wraps at corpus length)')
a=p.parse_args()
def read(path):
    values=array.array('f');values.frombytes(Path(path).read_bytes());return values
x,y=read(a.reference),read(a.candidate);b=a.batch;s=a.len*a.len
assert b>0 and a.len>0 and a.row_offset>=0
assert len(x)==len(y)==b*(2+2*s+3+6+s),(len(x),len(y))
assert all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y)

valid=[list(range(s)) for _ in range(b)]
if a.corpus:
    inputs=read(a.corpus);stride=s*22+19
    assert inputs and len(inputs)%stride==0,'bad corpus length: expected NHWC22 + global19 FP32 rows'
    num_rows=len(inputs)//stride
    for r in range(b):
        start=((a.row_offset+r)%num_rows)*stride
        mask=inputs[start:start+s*22:22]
        assert all(m in (0.0,1.0) for m in mask),'corpus channel zero must be a binary board mask'
        valid[r]=[xy for xy,m in enumerate(mask) if m==1.0]
        assert valid[r],'corpus row contains no valid board cells'
def metrics(u,v):
    errors=[abs(i-j) for i,j in zip(u,v)]
    assert len(u)==len(v) and len(u)>0
    return {'max_abs':max(errors),'rmse':math.sqrt(sum(d*d for d in errors)/len(errors)),'different':sum(d!=0 for d in errors),'count':len(errors)}
def softmax(v):
    ex=[math.exp(t-max(v)) for t in v];s=sum(ex);return [t/s for t in ex]
result={};offset=0;parts={}
for name,count in [('pass',2*b),('policy',2*b*s),('value',3*b),('score',6*b),('ownership',b*s)]:
    u=x[offset:offset+count];v=y[offset:offset+count];parts[name]=(u,v);result[name]=metrics(u,v);offset+=count
for channel in range(2):
    probs=[[],[]];tops=[]
    for r in range(b):
        pair=[]
        for q in range(2):
            logits=[parts['policy'][q][(r*s+xy)*2+channel] for xy in valid[r]]+[parts['pass'][q][2*r+channel]]
            pp=softmax(logits);probs[q].extend(pp);pair.append(max(range(len(pp)),key=pp.__getitem__))
        tops.append(pair[0]==pair[1])
    result[f'policy{channel}_prob']=metrics(*probs);result[f'policy{channel}_prob']['top1_agreement']=sum(tops)/b
    result[f'policy{channel}_prob']['top1_matches']=sum(tops)
    result[f'policy{channel}_prob']['top1_count']=b
result['ownership_valid']=metrics(*[[parts['ownership'][q][r*s+xy] for r in range(b) for xy in valid[r]] for q in range(2)])
values=[[],[]]
for q in range(2):
    for r in range(b):values[q].extend(softmax(parts['value'][q][3*r:3*r+3]))
result['value_prob']=metrics(*values)
print(json.dumps(result,indent=2))
