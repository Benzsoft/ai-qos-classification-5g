"""Bounded endpoint-conversation features for a development pilot, not 5-tuple flows."""
from pathlib import Path
import hashlib, hmac, json, re, zipfile
import numpy as np
import pandas as pd

VERSION=1
CONFIG={'version':VERSION,'sequence_length':30,'idle_seconds':30.0,'max_samples_per_capture':2000,
        'max_endpoint_pairs':100000,'seed':42,
        'unit':'first 30 packets of a bidirectional endpoint-pair inactivity segment',
        'features':['packet_length_bytes','interarrival_ms','direction_from_first_sender'],
        'protocol':'raw dissector label; categorical encoding deferred to train-only fitting',
        'prefix_lengths':[5,10,20,30]}
PORTS=re.compile(r'^\s*(\d{1,5})\s*(?:→|->|>)\s*(\d{1,5})(?:\s|$)')

def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def build_capture(archive,row,output_root,salt,config=None):
    cfg=dict(CONFIG if config is None else config)
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    dest=root/(row['capture_id']+'.npz');meta=dest.with_suffix('.json')
    signature=hashlib.sha256(json.dumps({'config':cfg,'row':row,'salt_digest':hashlib.sha256(salt).hexdigest()},sort_keys=True).encode()).hexdigest()
    if sha_file(archive)!=row['archive_sha256']:raise ValueError('Archive checksum mismatch')
    if meta.exists() and dest.exists():
        old=json.loads(meta.read_text())
        if old.get('signature')==signature and old.get('npz_sha256')==sha_file(dest):return old
    rng=np.random.default_rng(cfg['seed']+int(row['capture_id'][:8],16))
    buffers={};samples=[];eligible=0;total=invalid=port_hints=0;segments=0
    last_time=None
    def token(ip):return hmac.new(salt,ip.encode(),hashlib.sha256).hexdigest()
    required=['Time','Source','Destination','Length','Protocol','Info']
    with zipfile.ZipFile(archive) as z:
        members=[m for m in z.infolist() if not m.is_dir()]
        if len(members)!=1:raise ValueError('Expected one capture member')
        with z.open(members[0]) as f:
            for chunk in pd.read_csv(f,chunksize=50000,dtype=str,keep_default_na=False,encoding='utf-8',encoding_errors='backslashreplace'):
                if not set(required)<=set(chunk):raise ValueError('Missing packet columns')
                times=pd.to_datetime(chunk['Time'],errors='coerce')
                lengths=pd.to_numeric(chunk['Length'],errors='coerce')
                for values,t,length in zip(chunk[required].itertuples(index=False,name=None),times,lengths):
                    total+=1
                    _,src,dst,_,proto,info=values
                    if pd.isna(t) or pd.isna(length) or not np.isfinite(length) or length<=0 or not src or not dst or not proto:
                        invalid+=1;continue
                    if last_time is not None and t<last_time:raise ValueError('Timestamp order changed')
                    last_time=t
                    match=PORTS.match(info)
                    if match and all(0<=int(v)<=65535 for v in match.groups()):port_hints+=1
                    key=tuple(sorted((src,dst)))
                    state=buffers.get(key)
                    if state is None or (t-state['last']).total_seconds()>cfg['idle_seconds']:
                        if state is None and len(buffers)>=cfg['max_endpoint_pairs']:
                            raise ValueError('Endpoint-pair cap reached; no silent eviction')
                        segments+=1
                        state={'last':t,'first':src,'other':dst,'packets':[], 'protocols':[], 'done':False,'id':segments}
                        buffers[key]=state
                    gap=(t-state['last']).total_seconds()*1000
                    state['last']=t
                    if state['done']:continue
                    direction=0 if src==dst else (1 if src==state['first'] else -1)
                    state['packets'].append([float(length),float(gap),float(direction)])
                    state['protocols'].append(proto)
                    if len(state['packets'])==cfg['sequence_length']:
                        eligible+=1
                        entry=(np.array(state['packets'],dtype=np.float32),list(state['protocols']),
                               [token(state['first']),token(state['other'])],f"{row['capture_id']}:{state['id']}")
                        if len(samples)<cfg['max_samples_per_capture']:samples.append(entry)
                        else:
                            idx=int(rng.integers(eligible))
                            if idx<len(samples):samples[idx]=entry
                        state['done']=True;state['packets']=[];state['protocols']=[]
    if 'rows' in row and total!=row['rows']:
        raise ValueError('Feature scan row count differs from verified capture scan')
    # Sort retained sample IDs for stable artifact order.
    samples.sort(key=lambda x:x[3])
    n=len(samples);L=cfg['sequence_length']
    X=np.stack([s[0] for s in samples]) if n else np.empty((0,L,3),np.float32)
    protocols=np.array([s[1] for s in samples],dtype=str).reshape(n,L)
    endpoints=np.array([s[2] for s in samples],dtype='U64').reshape(n,2)
    ids=np.array([s[3] for s in samples],dtype=str)
    tmp=dest.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,X=X,protocols=protocols,endpoints=endpoints,sample_ids=ids)
    tmp.replace(dest)
    result={'capture_id':row['capture_id'],'application':row['application'],'split':row['split'],
            'recording_group':row['recording_group'],'source_file':row['source_file'],
            'signature':signature,'npz_sha256':sha_file(dest),'packet_rows_read':total,
            'invalid_feature_rows_skipped':invalid,'rows_with_leading_port_hint':port_hints,
            'port_hint_fraction_of_valid_rows':port_hints/max(1,total-invalid),
            'endpoint_pairs':len(buffers),'inactivity_segments':segments,'eligible_30_packet_segments':eligible,
            'saved_samples':n,'config':cfg,'feature_shape':list(X.shape),
            'sampling':'seeded uniform reservoir over eligible segments per capture',
            'limitations':['Endpoint pair may multiplex several transport flows; no 5-tuple claim.',
                           'Inactivity boundary is not a verified flow start.',
                           'Short segments below 30 packets excluded from every prefix comparison.',
                           'IP HMAC tokens are pseudonymous, not guaranteed anonymous; keep artifacts private.',
                           'Port hints are diagnostic only and are not used as flow keys.']}
    meta.write_text(json.dumps(result,indent=2));return result
