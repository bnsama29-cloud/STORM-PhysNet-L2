import numpy as np, json
from scipy import stats
from statsmodels.stats.multitest import multipletests
# Usage: run from a folder holding res/out_{A,B,C} (main predictions) and fma/fma_{A,B,C} (fixed_ma_control.zip)
S=range(42,52); B='ABC'
def load(fam,b,s,key='pred'):
    d='fma' if 'fixed_ma' in fam else 'res'
    sub=f'fma/fma_{b}' if d=='fma' else f'res/out_{b}'
    return np.load(f'{sub}/preds_{b}_{fam}_s{s}.npz')[key]
Y={b:np.load(f'res/out_{b}/meta_{b}.npz')['y'] for b in B}
def pe(fam,b,s,key='pred'):
    p=load(fam,b,s,key); y=Y[b]; return 1-((p-y)**2).mean(0)/y.var(0)
keys=['pred','pred_noise0.5','pred_noise1','pred_noise2','pred_out6','pred_out24','pred_out72','pred_lag3']
out={}
for fam in ['storm_fixed_ma','tf_res_fixed_ma']:
    for k in keys:
        a=np.array([[pe(fam,b,s,k) for s in S] for b in B]); out[f'{fam}|{k}']=a.mean((0,1)).round(4).tolist()
        print(fam,k,a.mean((0,1)).round(4), 'blockA' if k=='pred' else '', a[0].mean(0).round(3) if k=='pred' else '')
# paired tests clean + noise (seed means)
comps=[('storm_bz','storm_fixed_ma'),('tf_ffn_res','tf_res_fixed_ma'),('storm_bz','tf_res_fixed_ma'),('storm_fixed_ma','tf_res_fixed_ma')]
rows=[];ps=[]
for k in ['pred','pred_noise1','pred_noise2']:
    for a_,c in comps:
        d=np.array([[pe(a_,b,s,k)-pe(c,b,s,k) for s in S] for b in B])
        for h in range(3):
            sm=d[:,:,h].mean(0); p=stats.ttest_1samp(sm,0).pvalue; ps.append(p)
            per=[stats.ttest_1samp(d[i,:,h],0).pvalue for i in range(3)]
            nb=sum(min(1,3*q)<0.05 and np.sign(d[i,:,h].mean())==np.sign(sm.mean()) for i,q in enumerate(per))
            rows.append((k,a_,c,h,sm.mean(),p,nb))
holm=multipletests(ps,method='holm')[1]
for r,hp in zip(rows,holm): print('%-12s %-15s - %-16s h%d  %+.4f  p=%.1e holm=%.1e blocks=%d'%(r[0],r[1],r[2],r[3],r[4],r[5],hp,r[6]))
json.dump(out,open('fma_numbers.json','w'),indent=1)
# epochs, params
import glob
for f in glob.glob('fma/fma_*/runlog_*.jsonl'):
    L=[json.loads(l) for l in open(f)]
    for fam in ['storm_fixed_ma','tf_res_fixed_ma']:
        e=[x['epochs'] for x in L if x['family']==fam]; print(f[-7],fam,len(e),'runs, params',{x['n_params'] for x in L if x['family']==fam})
