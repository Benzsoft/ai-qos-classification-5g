"""Candidate relationships, not proof of session independence. Never assigns splits."""
from pathlib import Path
import collections,hashlib,itertools,json,zipfile
import numpy as np
import pandas as pd

VERSION=1
POLICY={'version':VERSION,'same_calendar_date_grouping':True,'row_match_min':20,
        'relative_row_match_min':100,'relative_row_containment_min':0.5,
        'shingle_seconds':10,'shared_distinct_shingles_min':5,'max_duration_seconds':86400}

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def hashes(frame):
 return set(pd.util.hash_pandas_object(frame,index=False).to_numpy(dtype='uint64').tolist())

def shingles(values,width=10):
 # Skip low-information constant/near-constant windows to reduce trivial matches.
 result=set()
 for i in range(len(values)-width+1):
  w=np.asarray(values[i:i+width],dtype='<i8')
  if np.count_nonzero(w)<3 or len(np.unique(w))<4:continue
  result.add(hashlib.sha256(w.tobytes()).digest())
 return result

def describe(df):
 required=['Time','Source','Destination','Protocol','Length']
 if not set(required)<=set(df):raise ValueError('Required columns missing')
 t=pd.to_datetime(df['Time'],errors='coerce')
 v=pd.to_numeric(df['Length'],errors='coerce')
 if t.isna().any() or not np.isfinite(v).all() or (v<0).any() or (v%1!=0).any():raise ValueError('Invalid time or byte count')
 if df[['Source','Destination','Protocol']].isna().any().any():raise ValueError('Missing endpoint/protocol')
 start,end=t.min(),t.max();duration=int((end-start).total_seconds())+1
 if duration>POLICY['max_duration_seconds']:raise ValueError('Capture exceeds bounded duration')
 frame=df[required].copy();frame['Time']=t.astype('int64');frame['Length']=v.astype('int64')
 relative=frame.copy();relative['Time']-=int(start.value)
 series=pd.Series(v.to_numpy(dtype='int64'),index=t).groupby(level=0).sum()
 grid=pd.date_range(start,end,freq='s');values=series.reindex(grid,fill_value=0).to_numpy(dtype='int64')
 if (t.dt.floor('s')!=t).any():raise ValueError('Expected integer-second aggregates')
 return {'start':str(start),'end':str(end),'dates':set(pd.date_range(start.normalize(),end.normalize(),freq='D').strftime('%Y-%m-%d')),
         'rows':len(df),'row_hashes':hashes(frame),'relative_hashes':hashes(relative),
         'endpoints':set(df.Source.astype(str))|set(df.Destination.astype(str)),
         'shingles':shingles(values,POLICY['shingle_seconds']),'duration_seconds':duration,
         'unique_seconds':int(t.nunique()),'out_of_order_rows':int((t.diff().dt.total_seconds()<0).sum())}

def relationship(a,b):
 common=len(a['row_hashes']&b['row_hashes']);rel=len(a['relative_hashes']&b['relative_hashes'])
 containment=rel/max(1,min(len(a['relative_hashes']),len(b['relative_hashes'])))
 shared=len(a['shingles']&b['shingles']);dates=sorted(a['dates']&b['dates'])
 reasons=[]
 if dates:reasons.append('same_calendar_date_proxy')
 if common>=POLICY['row_match_min']:reasons.append('shared_absolute_record_fingerprints')
 if rel>=POLICY['relative_row_match_min'] and containment>=POLICY['relative_row_containment_min']:reasons.append('shared_time_shifted_record_fingerprints')
 if shared>=POLICY['shared_distinct_shingles_min']:reasons.append('shared_variable_byte_sequence_shingles')
 return {'reasons':reasons,'shared_dates':dates,'shared_absolute_records':common,'shared_relative_records':rel,
         'relative_containment':containment,'shared_distinct_byte_shingles':shared,
         'endpoint_intersection_count':len(a['endpoints']&b['endpoints'])}

def run(root):
 root=Path(root);out=root/'relationships_v1';out.mkdir(exist_ok=True)
 audit_path=root/'gavist_full_audit.json';audit=json.loads(audit_path.read_text())
 reports=audit['reports']
 if audit.get('captures_scanned')!=122 or len(reports)!=122:raise ValueError('Expected complete 122-file audit')
 desc={}
 with zipfile.ZipFile(root/'gavist5g_v1.zip') as z:
  for row in reports:
   name=row['file'];member=z.getinfo(name)
   if member.file_size!=row['bytes']:raise ValueError('File size changed')
   h=hashlib.sha256()
   with z.open(member) as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
   if h.hexdigest()!=row['sha256']:raise ValueError('File checksum changed: '+name)
   with z.open(member) as f:df=pd.read_csv(f)
   if len(df)!=row['rows']:raise ValueError('Row count changed')
   desc[name]=describe(df);print('Audited',name,flush=True)
 names=sorted(desc);parent={n:n for n in names}
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 edges=[]
 for a,b in itertools.combinations(names,2):
  r=relationship(desc[a],desc[b])
  if r['reasons']:
   union(a,b);edges.append({'file_a':a,'file_b':b,**r})
 groups=collections.defaultdict(list)
 for n in names:groups[find(n)].append(n)
 assignments=[];group_rows=[]
 for members in groups.values():
  gid=hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()[:16]
  group_rows.append({'group_id':gid,'files':sorted(members),'file_count':len(members)})
  for name in members:assignments.append({'file':name,'candidate_group':gid,'split':'unassigned'})
 nondate=[e for e in edges if any(r!='same_calendar_date_proxy' for r in e['reasons'])]
 result={'stage':'gavist_capture_relationships','policy':POLICY,'source_audit_sha256':digest(audit_path),
         'captures_checked':len(names),'candidate_group_count':len(groups),'largest_group_files':max(map(len,groups.values())),
         'candidate_pair_count':len(edges),'content_candidate_pair_count':len(nondate),'content_candidates':nondate,
         'groups':group_rows,'training_ready':False,'publication_ready':False,
         'limitations':['Candidates are conservative grouping evidence, not proof of duplication or independence.',
         'Same-date grouping is global and may overgroup unrelated sessions; timezone provenance is unknown.',
         'Row matching excludes added geography, Org, ARTT, and Connection fields; 64-bit collisions remain possible.',
         'Byte shingles detect exact variable 10-second patterns only; no-match does not rule out transformed/partial reuse.',
         'Endpoint overlap alone does not trigger grouping; shared servers are normal.',
         'No raw IPs or fingerprint sets exported. No train/test split assigned.']}
 (out/'relationship_summary.json').write_text(json.dumps(result,indent=2))
 (out/'candidate_pairs.json').write_text(json.dumps(edges,indent=2))
 (out/'candidate_groups.json').write_text(json.dumps(assignments,indent=2))
 return result
