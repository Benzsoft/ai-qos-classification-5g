"""Deterministic development splits by capture file; not proof of independent sessions."""
from pathlib import PurePosixPath
import hashlib, json, math
from collections import defaultdict

APPLICATIONS = ('Zepeto', 'Teamfight_Tactics', 'YouTube_Live')

def build_plan(files, seed=42, applications=APPLICATIONS):
    grouped, seen = defaultdict(list), set()
    for f in files:
        name = f['name']
        if name in seen:
            raise ValueError('Duplicate inventory filename: '+name)
        seen.add(name)
        parts = PurePosixPath(name).parts
        if len(parts)<3 or not name.lower().endswith('.csv'): continue
        grouped[parts[-2]].append({'source_file':name, 'source_bytes':int(f['bytes']),
                                  'application':parts[-2], 'category':parts[-3]})
    overview = [{'application':a, 'capture_files':len(fs),
                 'source_bytes':sum(f['source_bytes'] for f in fs),
                 'eligible_for_three_way_development_split':len(fs)>=5}
                for a,fs in sorted(grouped.items())]
    rows=[]
    for app in applications:
        fs=grouped.get(app, [])
        if len(fs)<5: raise ValueError(f'{app}: need at least five files; found {len(fs)}')
        # Hash ranking is stable across inventory order and Python versions.
        ranked=sorted(fs, key=lambda f:hashlib.sha256(f"{seed}:{f['source_file']}".encode()).hexdigest())
        n_test=max(1, math.floor(len(fs)*.2))
        n_val=max(1, math.floor(len(fs)*.2))
        for i,f in enumerate(ranked):
            split='test' if i<n_test else ('validation' if i<n_test+n_val else 'train')
            rows.append(dict(f, split=split, capture_id=hashlib.sha256(f['source_file'].encode()).hexdigest()[:16],
                             session_independence='unverified', label_provenance='application directory',
                             acquisition_status='not_verified', timestamp_timezone='unknown'))
    counts={a:{s:sum(r['application']==a and r['split']==s for r in rows)
               for s in ['train','validation','test']} for a in applications}
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    summary={'stage':'capture_file_development_manifest','seed':seed,'manifest_sha256':digest,
             'selected_applications':list(applications),'capture_counts':counts,
             'capture_files':len(rows),'declared_uncompressed_bytes':sum(r['source_bytes'] for r in rows),
             'split_strategy':'within-application deterministic hash ranking of entire filenames; approximately 60/20/20',
             'file_overlap_between_partitions':False,'training_ready':False,
             'publication_ready':False,
             'limitations':['File disjointness does not establish independent sessions or devices.',
                            'No chronological or unseen-device generalization claim from this split.',
                            'Selected applications are a development subset; one application per category confounds category and application.',
                            'License and packet-level attribution remain unresolved.',
                            'Do not use the previous one-capture samples as independent train/test data.'],
             'required_before_training':['Acquire selected captures with hashes and version metadata.',
                                        'Identify recording dates, device/session provenance and possible file continuations.',
                                        'Group related captures into the SAME split and regenerate manifest if necessary.',
                                        'Detect exact/overlapping packet segments across files; review labels and flow extraction.',
                                        'Fit encoders and scalers on training only; freeze final split before tuning.']}
    return overview, rows, summary
