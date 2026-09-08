"""Summarize extraction counts without reading raw packets or endpoint tokens."""
import json
from pathlib import Path

def audit(root,out):
    summary=json.loads((Path(root)/'feature_summary.json').read_text())
    rows=[]
    for r in summary['reports']:
        n=r['inactivity_segments']; eligible=r['eligible_30_packet_segments']
        rows.append({k:r[k] for k in ['capture_id','application','split','recording_group','packet_rows_read','invalid_feature_rows_skipped','endpoint_pairs','inactivity_segments','eligible_30_packet_segments','saved_samples','port_hint_fraction_of_valid_rows']})
        rows[-1].update(short_segments=n-eligible,reservoir_dropped=eligible-r['saved_samples'])
    result={'stage':'extraction_diagnosis','captures':rows,'final_test_available':False,
      'unresolved':['No untouched final test reserved.','YouTube training has one recording-date group.','License and session/device provenance unresolved.','Endpoint segments may multiplex transport flows.','Port-hint differences require packet export/protocol review, not causal attribution.']}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'extraction_diagnosis.json').write_text(json.dumps(result,indent=2))
    return result
