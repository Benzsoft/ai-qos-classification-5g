"""GAViST5G ongoing-traffic windows. No model fitting or final-test scoring."""
from pathlib import Path
import hashlib,hmac,json,os,zipfile
import numpy as np
import pandas as pd

BUDGETS=[5,10,20,30]
EXPECTED_SPLIT="c6ee0b245ab3a317f212e77f627c2bc6b362760374adbbe2cdfbfb5ea31921f1"
CONFIG={"version":1,"window_seconds":30,"prefix_seconds":BUDGETS,
 "features":["summed_bytes","active_bidirectional_endpoint_pairs","aggregate_record_count"],
 "unit":"non-overlapping 30-second capture-aligned window; ongoing traffic",
 "empty_bins":"zero filled; missing seconds cannot be distinguished from inactivity",
 "endpoint_ablation":"largest-byte endpoint pair within each prefix; pseudonymous tokens",
 "scaling":"none; fit later on training only"}

def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()

def extract(df,key):
 required=["Time","Source","Destination","Length"]
 if not set(required)<=set(df) or df[required].isna().any().any():
  raise ValueError("Missing required fields")
 t=pd.to_datetime(df.Time,errors="raise")
 length=pd.to_numeric(df.Length,errors="raise")
 if not len(df) or not np.isfinite(length).all() or (length<0).any() or (length%1!=0).any():
  raise ValueError("Invalid summed byte counts")
 if (t.dt.floor("s")!=t).any():raise ValueError("Expected integer-second timestamps")
 start=t.min();span=int((t.max()-start).total_seconds())+1
 if span>86400:raise ValueError("Capture exceeds bounded duration")
 n=span//30;used=n*30
 seconds=((t-start).dt.total_seconds()).astype("int64")
 # Unordered endpoint pairs: no unverified uplink/downlink designation.
 pairs=[tuple(sorted((str(a),str(b)))) for a,b in zip(df.Source,df.Destination)]
 work=pd.DataFrame({"second":seconds.to_numpy(),"bytes":length.to_numpy(dtype="int64"),"pair":pairs})
 work=work[work.second<used].copy()
 numeric=np.zeros((used,3),dtype=np.float64)
 if len(work):
  totals=work.groupby("second").agg(byte_sum=("bytes","sum"),pair_count=("pair","nunique"),record_count=("bytes","size"))
  numeric[totals.index.to_numpy(dtype=int)]=totals.to_numpy(dtype=np.float64)
 X=numeric.reshape(n,30,3)
 endpoints=np.full((n,len(BUDGETS),2),"",dtype="U64")
 def token(v):return hmac.new(key,v.encode(),hashlib.sha256).hexdigest()
 work["window"]=work.second//30
 for w,part in work.groupby("window"):
  for j,budget in enumerate(BUDGETS):
   prefix=part[part.second < int(w)*30+budget]
   if prefix.empty:continue
   counts=prefix.groupby("pair")["bytes"].sum()
   # Tie handling independent of pandas group iteration order.
   candidates=list(counts.items())
   best=sorted(candidates,key=lambda item:(-int(item[1]),item[0]))[0][0]
   endpoints[int(w),j]=[token(best[0]),token(best[1])]
 return X,endpoints,{"span_seconds":span,"complete_windows":n,
  "discarded_tail_seconds":span-used,"discarded_tail_rows":int((seconds>=used).sum()),
  "zero_record_bins":int((numeric[:,2]==0).sum()),
  "empty_windows":int((X[:,:,2].sum(axis=1)==0).sum())}

def self_check():
 def make(t,length,src="a",dst="b"):
  return {"Time":pd.Timestamp("2024-01-01")+pd.Timedelta(seconds=t),
          "Source":src,"Destination":dst,"Length":length}
 df=pd.DataFrame([make(0,10),make(0,20,"b","a"),make(4,7),make(29,9)])
 x,e,meta=extract(df,b"x"*32)
 assert x.shape==(1,30,3)
 np.testing.assert_array_equal(x[0,0],[30,1,2])
 np.testing.assert_array_equal(x[0,1],[0,0,0])
 late=pd.concat([df,pd.DataFrame([make(20,10000,"future","server")])],ignore_index=True)
 x2,e2,_=extract(late,b"x"*32)
 np.testing.assert_array_equal(x[:,:10],x2[:,:10])
 np.testing.assert_array_equal(e[:,:2],e2[:,:2])
 assert not np.array_equal(e[:,3],e2[:,3])
 tail=pd.concat([df,pd.DataFrame([make(30,5)])],ignore_index=True)
 xt,_,mt=extract(tail,b"x"*32)
 assert len(xt)==1 and mt["discarded_tail_rows"]==1
 empty=pd.DataFrame([make(0,1),make(89,2)])
 xe,ee,me=extract(empty,b"x"*32)
 assert len(xe)==3 and me["empty_windows"]==1 and (ee[1]=="").all()
 print("Feature self-checks passed: binning, direction-independent pairs, prefix isolation, tails, empty windows.")

def run(root):
 root=Path(root);splitdir=root/"verified_split_v1"
 manifest=json.loads((splitdir/"frozen_manifest.json").read_text())
 summary=json.loads((splitdir/"verified_split_summary.json").read_text())
 ap=root/"gavist_full_audit.json";rp=root/"relationships_v1/relationship_summary.json"
 computed=hashlib.sha256(json.dumps({"audit":digest(ap),"relationships":digest(rp),
  "policy":summary["policy"],"manifest":manifest},sort_keys=True).encode()).hexdigest()
 if computed!=summary["signature"] or computed!=EXPECTED_SPLIT:
  raise ValueError("Frozen split signature mismatch; do not silently regenerate splits")
 if len(manifest)!=102 or len({r["file"] for r in manifest})!=102:
  raise ValueError("Expected 102 unique retained captures")
 groups={}
 for r in manifest:
  if r["split"] not in ["train","validation","test"]:raise ValueError("Invalid split")
  if groups.setdefault(r["group"],r["split"])!=r["split"]:raise ValueError("Group leakage")
 out=root/"features_windows_v1";out.mkdir(exist_ok=True)
 (out/"captures").mkdir(exist_ok=True)
 keypath=out/"endpoint_hmac_key.bin"
 if not keypath.exists():
  with keypath.open("xb") as f:f.write(os.urandom(32))
 key=keypath.read_bytes()
 if len(key)!=32:raise ValueError("Invalid endpoint key")
 config_sig=hashlib.sha256(json.dumps({"config":CONFIG,"split":computed,
  "key_digest":hashlib.sha256(key).hexdigest()},sort_keys=True).encode()).hexdigest()
 reports=[]
 with zipfile.ZipFile(root/"gavist5g_v1.zip") as z:
  for row in manifest:
   capture_id=hashlib.sha256(row["file"].encode()).hexdigest()[:20]
   npz=out/"captures"/(capture_id+".npz");meta=npz.with_suffix(".json")
   if npz.exists() and meta.exists():
    old=json.loads(meta.read_text())
    if old["config_signature"]==config_sig and old["npz_sha256"]==digest(npz):
     reports.append(old);print("Reused",capture_id,flush=True);continue
   h=hashlib.sha256()
   with z.open(row["file"]) as f:
    for b in iter(lambda:f.read(1048576),b""):h.update(b)
   if h.hexdigest()!=row["sha256"]:raise ValueError("Archive capture checksum mismatch")
   with z.open(row["file"]) as f:df=pd.read_csv(f)
   if len(df)!=row["rows"]:raise ValueError("Capture row count changed")
   X,endpoints,stats=extract(df,key)
   ids=np.array([capture_id+":"+str(i) for i in range(len(X))],dtype=str)
   tmp=npz.with_suffix(".partial")
   with tmp.open("wb") as f:
    np.savez_compressed(f,X=X,endpoints_by_prefix=endpoints,sample_ids=ids,
                       start_offset_seconds=np.arange(len(X))*30,prefix_seconds=np.array(BUDGETS))
   tmp.replace(npz)
   record={**row,"capture_id":capture_id,"config_signature":config_sig,
    "npz_sha256":digest(npz),**stats}
   meta.write_text(json.dumps(record,indent=2));reports.append(record)
   print("Extracted",capture_id,len(X),"windows",flush=True)
 coverage={}
 for split in ["train","validation","test"]:
  coverage[split]={}
  for label in sorted({r["label"] for r in manifest}):
   rr=[r for r in reports if r["split"]==split and r["label"]==label]
   coverage[split][label]={"files":len(rr),"groups_with_windows":len({r["group"] for r in rr if r["complete_windows"]>0}),
     "windows":sum(r["complete_windows"] for r in rr),"empty_windows":sum(r["empty_windows"] for r in rr)}
 ready=all(v["groups_with_windows"]>=2 and v["windows"]>0 for c in coverage.values() for v in c.values())
 result={"stage":"gavist_ongoing_window_features","split_signature":computed,"config":CONFIG,
  "captures_completed":len(reports),"coverage":coverage,"reports":reports,
  "coverage_gate_passed":ready,"training_ready":False,"publication_ready":False,
  "next_gate":"Review feature coverage before training; no preprocessing fitted or test predictions made.",
  "limitations":["Windows from the same capture remain correlated; count recording groups, not windows, as independent units.",
  "All prefixes use the same complete-30-second cohort; incomplete tails excluded.",
  "Zero-filled gaps may reflect inactivity or missing capture data; cannot distinguish from this source.",
  "Endpoint identities are extra information for an ablation, not main numerical features.",
  "Aggregate record count is NOT packet count; multiple records per second reflect upstream grouping.",
  "Capture starts are not verified application starts; ongoing classification only.",
  "Grouping does not establish independent devices or sessions; no geographic-shift claim.",
  "ARTT, Connection, geography, Org, Protocol, and absolute timestamps are not numeric predictors."]}
 (out/"source_manifest.json").write_text(json.dumps(manifest,indent=2))
 (out/"window_feature_summary.json").write_text(json.dumps(result,indent=2))
 return result
