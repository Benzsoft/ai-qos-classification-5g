"""Direct overlap verification and group-disjoint six-class split; no model fitting."""
from pathlib import Path
from collections import Counter
import hashlib,json,zipfile
import numpy as np
import pandas as pd

SEED=20260909
POLICY={'seed':SEED,'trials':5000,'group_fractions':[.6,.2,.2],
        'minimum_groups_per_class':{'train':2,'validation':2,'test':2},
        'task':'six-class aggregated traffic classification; not packet-prefix classification'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(name):
 if 'YouTube' in name:return 'YouTube'
 for token,value in [('/LOL/','League_of_Legends'),('/TFT/','Teamfight_Tactics'),('/VAL/','Valorant'),('/Netflix/','Netflix'),('/PrimeVideo/','Prime_Video'),('/Crave/','Crave')]:
  if token in name:return value
 raise ValueError('Unrecognized application path: '+name)

def records(df):
 # Direct tuple comparison, preserving multiplicity, with no fingerprint matching.
 cols=['Time','Source','Destination','Protocol','Length']
 if df[cols].isna().any().any():raise ValueError('Missing comparison fields')
 t=pd.to_datetime(df.Time,errors='raise');v=pd.to_numeric(df.Length,errors='raise')
 if not np.isfinite(v).all() or (v<0).any() or (v%1!=0).any():raise ValueError('Invalid bytes')
 return Counter(zip(t.astype('int64').tolist(),df.Source.astype(str),df.Destination.astype(str),df.Protocol.astype(str),v.astype('int64').tolist()))

def plan(audit,relationships,excluded):
 lookup={r['file']:r for r in audit['reports']};groups=[];seen=set()
 for g in relationships['groups']:
  for f in g['files']:
   if f in seen or f not in lookup:raise ValueError('Invalid group membership')
   seen.add(f)
  files=sorted(set(g['files'])-set(excluded))
  if files:groups.append((g['group_id'],files))
 if seen!=set(lookup):raise ValueError('Missing group assignments')
 groups.sort();classes=sorted({label(f) for _,fs in groups for f in fs})
 if len(classes)!=6 or 'Crave' in classes:raise ValueError('Expected six retained classes')
 counts=np.array([[sum(label(f)==c for f in fs) for c in classes] for _,fs in groups])
 present=counts>0;totals=counts.sum(axis=0);n=len(groups)
 nval=max(1,round(.2*n));ntest=nval;ntrain=n-nval-ntest
 rng=np.random.default_rng(SEED);best=None
 for _ in range(POLICY['trials']):
  order=rng.permutation(n);parts=[order[:ntrain],order[ntrain:ntrain+nval],order[ntrain+nval:]]
  if any((present[p].sum(axis=0)<m).any() for p,m in zip(parts,[2,2,2])):continue
  score=sum(float(np.square(counts[p].sum(axis=0)/totals-f).sum()) for p,f in zip(parts,[.6,.2,.2]))
  if best is None or score<best[0]:best=(score,parts)
 if best is None:raise ValueError('No feasible group split; do not fall back to random rows')
 manifest=[];coverage={}
 for split,idx in zip(['train','validation','test'],best[1]):
  coverage[split]={}
  for c in classes:
   fs=[f for i in idx for f in groups[i][1] if label(f)==c]
   coverage[split][c]={'files':len(fs),'groups':sum(any(label(f)==c for f in groups[i][1]) for i in idx),'aggregate_rows':sum(lookup[f]['rows'] for f in fs)}
  for i in idx:
   gid,files=groups[i]
   for f in files:manifest.append({'file':f,'label':label(f),'group':gid,'split':split,'sha256':lookup[f]['sha256'],'rows':lookup[f]['rows']})
 return sorted(manifest,key=lambda r:r['file']),coverage

def save_once(path,text):
 if path.exists() and path.read_text()!=text:raise ValueError('Existing frozen artifact differs: '+str(path))
 path.write_text(text)

def run(root):
 root=Path(root);ap=root/'gavist_full_audit.json';rp=root/'relationships_v1/relationship_summary.json'
 audit=json.loads(ap.read_text());rel=json.loads(rp.read_text())
 if rel['source_audit_sha256']!=sha(ap):raise ValueError('Source audit changed')
 if len(audit['reports'])!=122 or rel['captures_checked']!=122:raise ValueError('Incomplete upstream audit')
 pairs=[p for p in rel['content_candidates'] if {label(p['file_a']),label(p['file_b'])}=={'Crave','YouTube'}]
 if len(pairs)!=10:raise ValueError('Expected exactly ten proposed cross-label pairs')
 excluded={f for p in pairs for f in [p['file_a'],p['file_b']]}
 if len(excluded)!=20:raise ValueError('Unexpected exclusion count')
 counters={}
 with zipfile.ZipFile(root/'gavist5g_v1.zip') as z:
  for r in audit['reports']:
   info=z.getinfo(r['file'])
   if info.file_size!=r['bytes']:raise ValueError('File size mismatch')
   h=hashlib.sha256()
   with z.open(info) as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
   if h.hexdigest()!=r['sha256']:raise ValueError('Archive content changed')
   if r['file'] in excluded:
    with z.open(info) as f:df=pd.read_csv(f)
    if len(df)!=r['rows']:raise ValueError('Row count mismatch')
    counters[r['file']]=records(df)
 verified=[]
 for p in pairs:
  a,b=p['file_a'],p['file_b']
  if label(a)!='Crave':a,b=b,a
  ca,cb=counters[a],counters[b];missing=sum((ca-cb).values())
  verified.append({'crave_file':a,'youtube_file':b,'crave_rows':sum(ca.values()),'matched_rows_with_multiplicity':sum((ca&cb).values()),'unmatched_crave_rows':missing})
 # If any fingerprint allegation fails direct comparison, preserve diagnostics and stop.
 out=root/'verified_split_v1';out.mkdir(exist_ok=True)
 (out/'direct_overlap_verification.json').write_text(json.dumps(verified,indent=2))
 if any(r['unmatched_crave_rows'] for r in verified):raise ValueError('Containment not fully confirmed; review verification report')
 manifest,coverage=plan(audit,rel,excluded)
 excluded_records=[{'file':f,'reason':'Unresolved cross-label containment; both captures excluded, originals preserved'} for f in sorted(excluded)]
 signature=hashlib.sha256(json.dumps({'audit':sha(ap),'relationships':sha(rp),'policy':POLICY,'manifest':manifest},sort_keys=True).encode()).hexdigest()
 summary={'stage':'gavist_verified_group_split','signature':signature,'policy':POLICY,'direct_verification':verified,
 'excluded_files':len(excluded),'retained_files':len(manifest),'retained_groups':len({r['group'] for r in manifest}),
 'retained_aggregate_rows':sum(r['rows'] for r in manifest),'coverage':coverage,'group_disjoint':True,
 'split_ready':True,'training_ready':False,'publication_ready':False,
 'limitations':['Labels are file-derived and not independently verified.','Remaining candidate groups are conservative proxies, not proven independent sessions/devices.',
 'Split balances file-class coverage only; no model scores used. Not a geographic or chronological holdout.',
 'Final test must remain unused during feature/model selection.','Feature-window design and eligible sample counts still require validation.',
 'No early packet classification or measured latency claims; exclude ARTT and Connection.',
 'Removing all Crave files means six application classes; all six model families can still be retained.']}
 for name,value in [('frozen_manifest.json',manifest),('exclusions.json',excluded_records),('verified_split_summary.json',summary)]:save_once(out/name,json.dumps(value,indent=2))
 return summary
