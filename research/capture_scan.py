"""Resumable file acquisition and bounded-memory capture diagnostics, version 1."""
import hashlib, json, urllib.request, urllib.parse, zipfile, os
from pathlib import Path
import numpy as np
import pandas as pd

SCAN_VERSION=1

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def acquire_capture(row, root, cap=128*1024*1024):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    archive=root/(row['capture_id']+'.zip');receipt=archive.with_suffix('.receipt.json')
    if archive.exists() and receipt.exists():
        info=json.loads(receipt.read_text())
        if info['source_file']==row['source_file'] and info['sha256']==sha256_file(archive):
            return archive,info
    url='https://www.kaggle.com/api/v1/datasets/download/kimdaegyeom/5g-traffic-datasets/'+urllib.parse.quote(row['source_file'],safe='')+'?datasetVersionNumber=1'
    tmp=archive.with_suffix('.partial')
    try:
        with urllib.request.urlopen(url,timeout=60) as r,tmp.open('wb') as f:
            n=0
            while True:
                b=r.read(min(1024*1024,cap-n+1))
                if not b:break
                n+=len(b)
                if n>cap:raise ValueError('Archive exceeds 128 MiB cap')
                f.write(b)
        with zipfile.ZipFile(tmp) as z:
            members=[m for m in z.infolist() if not m.is_dir()]
            if len(members)!=1 or not members[0].filename.endswith('.csv'):
                raise ValueError('Expected one CSV member')
            if members[0].file_size!=row['source_bytes']:
                raise ValueError('Inventory size and archive member size differ')
        tmp.replace(archive)
    finally:tmp.unlink(missing_ok=True)
    info={'source_file':row['source_file'],'requested_version':1,'sha256':sha256_file(archive),'archive_bytes':archive.stat().st_size}
    receipt.write_text(json.dumps(info,indent=2));return archive,info

def scan_capture(archive,row,root,receipt):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    dest=root/(row['capture_id']+'.json');finger=root/(row['capture_id']+'.npy')
    if dest.exists() and finger.exists():
        cached=json.loads(dest.read_text())
        if cached.get('scan_version')==SCAN_VERSION and cached.get('archive_sha256')==receipt['sha256'] and cached.get('fingerprint_sha256')==sha256_file(finger):
            return dict(cached,split=row['split'])
    required=['Time','Source','Destination','Protocol','Length','Info']
    total=bad=backwards=missing=0;previous=None;start=end=None;hashes=[];protocols={}
    with zipfile.ZipFile(archive) as z:
        with z.open(z.infolist()[0]) as stream:
            for df in pd.read_csv(stream,chunksize=50000,dtype=str,keep_default_na=False):
                if not set(required)<=set(df):raise ValueError('Required packet columns missing')
                total+=len(df)
                if total>20_000_000:raise ValueError('Capture row cap exceeded')
                t=pd.to_datetime(df['Time'],errors='coerce');bad+=int(t.isna().sum())
                good=t.dropna()
                if len(good):
                    backwards+=int((good.diff().dt.total_seconds()<0).sum())
                    if previous is not None and good.iloc[0]<previous:backwards+=1
                    previous=good.iloc[-1]
                    start=good.min() if start is None else min(start,good.min())
                    end=good.max() if end is None else max(end,good.max())
                missing+=int(df[required].eq('').any(axis=1).sum())
                for k,v in df['Protocol'].value_counts().items():protocols[k]=protocols.get(k,0)+int(v)
                # Excludes frame number: duplicated packets can be renumbered in exported files.
                hashes.append(pd.util.hash_pandas_object(df[required],index=False).to_numpy(dtype=np.uint64))
    if not hashes:raise ValueError('Empty capture')
    unique=np.unique(np.concatenate(hashes));del hashes
    np.save(finger,unique,allow_pickle=False)
    result={'scan_version':SCAN_VERSION,**row,'archive_sha256':receipt['sha256'],
            'fingerprint_sha256':sha256_file(finger),'rows':total,'timestamp_parse_failures':bad,
            'timestamp_backsteps':backwards,'rows_with_empty_required_fields':missing,
            'start_time':str(start),'end_time':str(end),'timezone':'unknown',
            'protocol_counts':protocols,'within_file_repeated_fingerprints':total-len(unique),
            'fingerprint_note':'64-bit fingerprints identify candidate duplicates, not collision-free proof.',
            'flow_id_status':'not constructed; dedicated port columns absent','training_ready':False}
    dest.write_text(json.dumps(result,indent=2));return result

def compare_captures(reports,root):
    pairs=[]
    for i,a in enumerate(reports):
        ah=np.load(Path(root)/(a['capture_id']+'.npy'),mmap_mode='r',allow_pickle=False)
        for b in reports[i+1:]:
            bh=np.load(Path(root)/(b['capture_id']+'.npy'),mmap_mode='r',allow_pickle=False)
            n=int(np.intersect1d(ah,bh,assume_unique=True).size)
            time_overlap=False
            if a['start_time']!='None' and b['start_time']!='None':
                time_overlap=max(pd.Timestamp(a['start_time']),pd.Timestamp(b['start_time']))<=min(pd.Timestamp(a['end_time']),pd.Timestamp(b['end_time']))
            if n or time_overlap:
                pairs.append({'capture_a':a['capture_id'],'capture_b':b['capture_id'],
                              'cross_split':a['split']!=b['split'],'shared_packet_fingerprints':n,
                              'overlapping_naive_time_ranges':bool(time_overlap),
                              'interpretation':'candidate for review; clock timezone and shared session not established'})
    return pairs
