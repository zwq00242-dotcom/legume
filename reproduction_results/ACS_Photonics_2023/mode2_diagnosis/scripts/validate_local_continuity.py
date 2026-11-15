from _paths import BASE, REPO, CACHE
import sys
import compute_mode2_fullmap as m
import numpy as np,json
# The concrete edge that exposed the narrow avoided crossing in the first audit.
a=np.array([.06933333333333332,-.04266666666666667]);b=np.array([.07466666666666667,-.04266666666666667])
g=m.solve(np.linspace(a,b,65));s=m.state(g,0,2);rows=[]
for i in range(65):
 s,d=m.advance(s,g,i);rows.append(m.observables(g,i,s,d))
assert s['mode']==2
m.dumpcsv('data/local_continuity65.csv',rows)
f=np.array([r['frequency_real'] for r in rows]);q=np.array([r['Q'] for r in rows]);gap=np.array([r['nearest_frequency_gap'] for r in rows])
st=dict(points=65,mode_indices=sorted(set(r['mode_index'] for r in rows)),min_overlap=min(r['overlap'] for r in rows),max_frequency_step=float(max(abs(np.diff(f)))),max_log10_Q_step=float(max(abs(np.diff(np.log10(q))))),minimum_gap=float(min(gap)))
(m.OUT/'validation/local_continuity65.json').write_text(json.dumps(st,indent=2));print(st)
