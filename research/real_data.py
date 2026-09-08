"""Bounded acquisition of public Kaggle files for schema inspection."""
import hashlib, io, json, time, urllib.parse, urllib.request, zipfile
from pathlib import Path
import pandas as pd
BASE = 'https://www.kaggle.com/api/v1/datasets/'
DATASET = 'kimdaegyeom/5g-traffic-datasets'

def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('Metadata response exceeded limit')
    return json.loads(raw)

def inventory():
    metadata = get_json(BASE + 'view/' + DATASET)
    files, tokens = {}, set()
    token = None
    for _ in range(100):
        url = BASE + 'list/' + DATASET
        if token:
            url += '?' + urllib.parse.urlencode({'pageToken': token})
        page = get_json(url)
        for item in page.get('datasetFiles', []):
            name = item.get('name') or item.get('nameNullable')
            if name:
                files[name] = {'name': name, 'bytes': int(item.get('totalBytes', 0))}
        token = page.get('nextPageToken') or page.get('nextPageTokenNullable')
        if not token:
            return metadata, list(files.values())
        if token in tokens:
            raise RuntimeError('Repeated pagination token; refusing incomplete inventory')
        tokens.add(token)
    raise RuntimeError('Inventory page limit exceeded')

def acquire_preview(item, destination, rows=5000, max_download=64*1024*1024):
    """Download one bounded archive and parse only a prefix. No raw IPs printed."""
    name = item['name']
    if not name.endswith('.csv'):
        raise ValueError('Only CSV files supported')
    url = BASE + 'download/' + DATASET + '/' + urllib.parse.quote(name, safe='') + '?datasetVersionNumber=1'
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(name.encode()).hexdigest()[:12]
    archive = directory / (digest + '.download')
    temp = archive.with_suffix('.partial')
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp.open('wb') as target:
            size = 0
            while True:
                block = response.read(min(1024*1024, max_download-size+1))
                if not block: break
                size += len(block)
                if size > max_download:
                    raise ValueError('File exceeds 64 MiB download cap; choose a smaller source file')
                target.write(block)
        temp.replace(archive)
    finally:
        temp.unlink(missing_ok=True)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if not m.is_dir() and m.filename.endswith('.csv')]
            if len(members) != 1:
                raise ValueError('Expected single CSV in file download')
            member = members[0]
            if member.file_size > 2*1024**3:
                raise ValueError('Uncompressed file exceeds inspection limit')
            with z.open(member) as stream:
                df = pd.read_csv(stream, nrows=rows)
    else:
        df = pd.read_csv(archive, nrows=rows)
    sample = directory / (digest + '_prefix.csv')
    df.to_csv(sample, index=False)
    parts = Path(name).parts
    temporal = {}
    if 'Time' in df:
        parsed = pd.to_datetime(df['Time'], errors='coerce')
        temporal = {'parse_failures': int(parsed.isna().sum()),
                    'monotonic_in_file': bool(parsed.is_monotonic_increasing),
                    'prefix_span_seconds': float((parsed.max()-parsed.min()).total_seconds()) if parsed.notna().any() else None,
                    'timezone': 'not established; no cross-capture chronology assumed'}
    record = {
        'timestamp_inspection': temporal,
        'dataset': DATASET, 'requested_version':1, 'source_file':name,
        'source_declared_bytes':item['bytes'], 'download_bytes':archive.stat().st_size,
        'download_sha256':sha, 'prefix_csv_sha256':hashlib.sha256(sample.read_bytes()).hexdigest(),
        'rows_inspected':len(df), 'columns':list(df.columns),
        'dtypes':{c:str(df[c].dtype) for c in df},
        'missing':{c:int(df[c].isna().sum()) for c in df},
        'unique_counts':{c:int(df[c].nunique()) for c in df},
        'category_from_path':parts[-3] if len(parts)>=3 else None,
        'application_from_path':parts[-2] if len(parts)>=2 else None,
        'label_status':'capture-level path labels; not independently validated per packet',
        'sampling':'first rows only; schema inspection, not representative benchmark',
        'training_ready':False,
        'pending':['license verification','timestamp semantics','packet ordering','flow/session independence','label validation'],
    }
    (directory/(digest+'_report.json')).write_text(json.dumps(record,indent=2))
    return record
