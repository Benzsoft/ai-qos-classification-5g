"""Bounded CSV diagnostics. Never creates labels, sequences, or evaluation splits."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def audit_csv(path, output_root, *, label, provenance, max_rows=100_000):
    path = Path(path)
    if max_rows <= 0:
        raise ValueError('max_rows must be positive')
    df = pd.read_csv(path, nrows=max_rows + 1)
    truncated = len(df) > max_rows
    df = df.iloc[:max_rows].copy()
    if df.empty or label not in df:
        raise ValueError(f'Nonempty CSV with label column {label!r} required')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(output_root) / stamp
    out.mkdir(parents=True, exist_ok=False)
    # Hash the audited representation; do not scan a huge raw capture just to hash it.
    digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    schema = pd.DataFrame({'column': df.columns, 'dtype': [str(df[c].dtype) for c in df],
                           'missing': [int(df[c].isna().sum()) for c in df],
                           'unique_nonnull': [int(df[c].nunique()) for c in df]})
    counts = df[label].value_counts(dropna=False).rename_axis('label').reset_index(name='count')
    counts['fraction'] = counts['count'] / len(df)
    numeric = df.select_dtypes(include=np.number)
    warnings = ['Prefix audit only; no representative sampling or split validation performed.',
                'Identifiers and timestamps require semantic review, even if column names are present.']
    if provenance == 'repository_synthetic':
        warnings.extend(['Class-conditioned independent rows; not measured 5G traffic.',
                         'No genuine flow histories: do not train temporal models on adjacent CSV rows.',
                         'Not eligible for the real-data benchmark; use only for diagnostics/reproduction.'])
    features = df.drop(columns=[label])
    duplicate_features = features.duplicated(keep=False)
    conflicting = 0
    if duplicate_features.any():
        hashes = pd.util.hash_pandas_object(features, index=False)
        conflicting = int(df.groupby(hashes)[label].nunique().gt(1).sum())
    report = {'source_file': path.name, 'provenance': provenance,
              'rows_audited': len(df), 'prefix_truncated': truncated,
              'audited_csv_sha256': digest, 'raw_file_bytes': path.stat().st_size,
              'label_column': label, 'label_missing': int(df[label].isna().sum()),
              'duplicate_rows': int(df.duplicated().sum()),
              'conflicting_feature_groups': conflicting,
              'infinite_numeric_values': int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
              'warnings': warnings, 'training_ready': False}
    schema.to_csv(out / 'schema.csv', index=False)
    counts.to_csv(out / 'class_counts.csv', index=False)
    if len(numeric.columns):
        numeric.describe().to_csv(out / 'numeric_summary.csv')
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(8, 4))
    shown = counts.head(20)
    ax.bar(shown['label'].astype(str), shown['count'])
    ax.set(title=f'Data audit: {provenance} (top 20 labels maximum)', ylabel='Audited rows')
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig(out / 'class_counts.png', dpi=180)
    plt.close(fig)
    return report, out
