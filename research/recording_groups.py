"""Conservative calendar grouping for a fixed-setting development pilot."""
import datetime as dt
import hashlib
import json
from collections import defaultdict

def grouped_pilot(scan):
    reports=scan['reports']
    if scan.get('errors') or len(reports)!=24 or scan.get('completed_captures')!=24:
        raise ValueError('Complete the verified 24-capture scan first')
    ids=[r['capture_id'] for r in reports]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate capture ID')
    if any(r['timestamp_parse_failures'] or r['timestamp_backsteps'] for r in reports):
        raise ValueError('Timestamp problems require review')
    parent=list(range(len(reports)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    dates=[]
    for r in reports:
        start=dt.datetime.fromisoformat(r['start_time']).date()
        end=dt.datetime.fromisoformat(r['end_time']).date()
        if end<start or (end-start).days>31:raise ValueError('Unexpected capture date range')
        dates.append({start+dt.timedelta(days=x) for x in range((end-start).days+1)})
    # Global date grouping: same-day captures stay together even across applications.
    for i in range(len(reports)):
        for j in range(i):
            if dates[i]&dates[j]:union(i,j)
    # Conservatively join any explicit fingerprint/time-overlap candidates too.
    lookup={v:i for i,v in enumerate(ids)}
    for pair in scan.get('candidate_overlap_pairs',[]):
        union(lookup[pair['capture_a']],lookup[pair['capture_b']])
    members=defaultdict(list)
    for i in range(len(reports)):members[find(i)].append(i)
    groups=[]
    for indices in members.values():
        names=sorted(ids[i] for i in indices)
        groups.append({'group_id':hashlib.sha256('|'.join(names).encode()).hexdigest()[:16],
                       'indices':indices,'start':min(reports[i]['start_time'] for i in indices),
                       'end':max(reports[i]['end_time'] for i in indices)})
    by_app=defaultdict(list)
    for g in groups:
        for app in {reports[i]['application'] for i in g['indices']}:by_app[app].append(g)
    if any(len(gs)<2 for gs in by_app.values()):raise ValueError('At least two groups per application required')
    # Latest group per application held out; union across apps avoids group leakage.
    test_groups={max(gs,key=lambda g:(g['end'],g['group_id']))['group_id'] for gs in by_app.values()}
    manifest=[]
    for g in groups:
        split='test' if g['group_id'] in test_groups else 'train'
        for i in g['indices']:
            r=reports[i]
            manifest.append({k:r[k] for k in ['capture_id','source_file','application','category','rows','archive_sha256','start_time','end_time']} | {'recording_group':g['group_id'],'split':split})
    manifest.sort(key=lambda r:r['source_file'])
    counts={app:{split:{'captures':sum(r['application']==app and r['split']==split for r in manifest),
                                'groups':len({r['recording_group'] for r in manifest if r['application']==app and r['split']==split}),
                                'rows':sum(r['rows'] for r in manifest if r['application']==app and r['split']==split)}
                 for split in ['train','test']} for app in sorted(by_app)}
    if any(c['train']['groups']==0 for c in counts.values()):raise ValueError('Shared holdout leaves an application without training data')
    summary={'stage':'conservative_recording_group_pilot','capture_counts':counts,
             'captures':len(manifest),'groups':len(groups),
             'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
             'validation_partition':None,'protocol':'fixed-settings pilot; latest recording group held out per application',
             'model_families':['random_forest','svm','knn','lstm','bilstm','ip_embedding'],
             'license':scan.get('license'),'publication_ready':False,'feature_extraction_ready':True,
             'limitations':['Calendar grouping is a conservative proxy, not verified session/device identity.',
                            'Latest group per application plus shared-date holdouts; not a strictly chronological evaluation, even within each application.',
                            'No model tuning, early stopping, feature selection or threshold selection using test data.',
                            'No independent validation group for YouTube Live; fixed settings only.',
                            'Pilot test is development evidence and must not become the final repeatedly inspected publication test.',
                            'License, packet-level attribution and flow extraction still require review.']}
    return manifest,summary
