"""Validation-only, fixed 10-second six-model benchmark. Never loads test NPZs."""
from pathlib import Path
import os,sys,json,hashlib,time,platform
os.environ.setdefault("TF_DETERMINISTIC_OPS","1")
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
tf.config.experimental.enable_op_determinism()
for gpu in tf.config.list_physical_devices("GPU"):
 try:tf.config.experimental.set_memory_growth(gpu,True)
 except RuntimeError:pass

SEED=42
PREFIX=10
SPLIT="c6ee0b245ab3a317f212e77f627c2bc6b362760374adbbe2cdfbfb5ea31921f1"
NAMES=["random_forest","svm","knn","lstm","bilstm","ip_embedding"]
SETTINGS={"version":1,"seed":SEED,"prefix_seconds":PREFIX,"epochs":20,"batch_size":32,
 "rf_trees":100,"svm_C":1.0,"svm_gamma":"scale","knn_k":5,"recurrent_units":64,
 "dropout":0.3,"optimizer":"Adam","learning_rate":0.001,
 "preprocess":"log1p then training-only StandardScaler",
 "balancing":"training-only random oversampling to largest class",
 "empty_prefix":"no observed traffic; excluded from conditional model metrics",
 "repeatability":"each neural configuration trained twice; exact weights and validation probabilities",
 "selection":"fixed settings; validation reporting only; no final-test loading"}

def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()

def active_mask(X):
 return X[:,:PREFIX,2].sum(axis=1)>0

def load_data(root):
 root=Path(root);summary=json.loads((root/"window_feature_summary.json").read_text())
 if summary["split_signature"]!=SPLIT or summary["captures_completed"]!=102 or not summary["coverage_gate_passed"]:
  raise ValueError("Feature coverage/split gate failed")
 manifest=json.loads((root/"source_manifest.json").read_text())
 frozen=json.loads((root.parent/"verified_split_v1/frozen_manifest.json").read_text())
 if manifest!=frozen:raise ValueError("Feature manifest differs from frozen source")
 split_summary=json.loads((root.parent/"verified_split_v1/verified_split_summary.json").read_text())
 computed=hashlib.sha256(json.dumps({"audit":digest(root.parent/"gavist_full_audit.json"),
  "relationships":digest(root.parent/"relationships_v1/relationship_summary.json"),
  "policy":split_summary["policy"],"manifest":manifest},sort_keys=True).encode()).hexdigest()
 if computed!=SPLIT:raise ValueError("Frozen split integrity check failed")
 byfile={r["file"]:r for r in summary["reports"]}
 if len(byfile)!=102:raise ValueError("Duplicate/missing feature reports")
 groups={};seen=set();checks={}
 packets={s:{k:[] for k in ["X","endpoints","y","group","capture","ids"]} for s in ["train","validation"]}
 for row in manifest:
  if groups.setdefault(row["group"],row["split"])!=row["split"]:raise ValueError("Group leakage")
  if row["split"]=="test":continue # Absolutely no test feature loading.
  r=byfile[row["file"]]
  for key in ["split","label","group","sha256"]:
   if r[key]!=row[key]:raise ValueError("Feature report metadata mismatch")
  p=root/"captures"/(r["capture_id"]+".npz")
  if digest(p)!=r["npz_sha256"]:raise ValueError("Feature checksum mismatch")
  checks[r["capture_id"]]=r["npz_sha256"]
  with np.load(p,allow_pickle=False) as z:
   X=z["X"].copy();n=len(X)
   if X.shape!=(n,30,3) or not np.isfinite(X).all() or (X<0).any():raise ValueError("Invalid numerical features")
   budgets=z["prefix_seconds"].tolist()
   if budgets!=[5,10,20,30] or z["endpoints_by_prefix"].shape!=(n,4,2):raise ValueError("Invalid endpoint prefix array")
   endpoints=z["endpoints_by_prefix"][:,budgets.index(PREFIX),:].copy()
   ids=z["sample_ids"].copy()
   if len(ids)!=n:raise ValueError("Sample ID mismatch")
   for sid in ids.tolist():
    if sid in seen:raise ValueError("Duplicate sample ID")
    seen.add(sid)
   d=packets[row["split"]]
   for key,val in [("X",X),("endpoints",endpoints),("ids",ids),("y",np.full(n,row["label"])),
     ("group",np.full(n,row["group"])),("capture",np.full(n,r["capture_id"]))]:d[key].append(val)
 return {s:{k:np.concatenate(v) for k,v in d.items()} for s,d in packets.items()},checks

def prepare(data):
 labels=sorted(set(data["train"]["y"]));mapping={c:i for i,c in enumerate(labels)}
 if len(labels)!=6:raise ValueError("Expected six classes")
 train_active=active_mask(data["train"]["X"])
 scale=StandardScaler().fit(np.log1p(data["train"]["X"][train_active,:PREFIX,:]).reshape(-1,3))
 vocab={v:i+1 for i,v in enumerate(sorted(set(data["train"]["endpoints"][train_active].ravel())-{""}))}
 result={};coverage={}
 for split,d in data.items():
  mask=active_mask(d["X"])
  seq=scale.transform(np.log1p(d["X"][mask,:PREFIX,:]).reshape(-1,3)).reshape(-1,PREFIX,3).astype("float32")
  endpoints=np.array([[vocab.get(v,0) for v in pair] for pair in d["endpoints"][mask]],dtype=np.int32)
  y=np.array([mapping[c] for c in d["y"][mask]],dtype=np.int32)
  if set(y)!=set(range(6)):raise ValueError("A class has no active prefixes")
  result[split]={"seq":seq,"flat":seq.reshape(len(seq),-1),"endpoints":endpoints,"y":y,"mask":mask}
  coverage[split]={}
  for c in labels:
   allc=d["y"]==c;eligible=allc&mask
   count=len(set(d["group"][eligible]))
   coverage[split][c]={"all_windows":int(allc.sum()),"active_prefixes":int(eligible.sum()),
    "empty_prefixes":int((allc&~mask).sum()),"active_fraction":float(eligible.sum()/allc.sum()),
    "groups_with_active_prefixes":count}
   if count<2:raise ValueError("Fewer than two active groups for "+split+" / "+c)
 return result,{"labels":labels,"endpoint_vocab":vocab,"scaler":scale},coverage

def sample_indices(y,balanced):
 idx=np.arange(len(y))
 if balanced:
  rng=np.random.default_rng(SEED);target=max(np.bincount(y))
  idx=np.concatenate([np.concatenate([np.flatnonzero(y==c),
   rng.choice(np.flatnonzero(y==c),target-np.sum(y==c),replace=True)]) for c in sorted(set(y))])
  rng.shuffle(idx)
 return idx

@tf.keras.utils.register_keras_serializable(package="gavist")
class KnownEmbedding(tf.keras.layers.Layer):
 def __init__(self,count,width=8,**kwargs):
  super().__init__(**kwargs);self.count=count;self.width=width
  self.embedding=tf.keras.layers.Embedding(count,width)
 def build(self,input_shape):
  self.embedding.build(input_shape);super().build(input_shape)
 def call(self,ids):
  e=self.embedding(ids)
  return e*tf.cast(tf.expand_dims(ids!=0,-1),e.dtype)
 def get_config(self):
  return {**super().get_config(),"count":self.count,"width":self.width}

def build_model(name,vocab_size):
 L=tf.keras.layers;seq=L.Input(shape=(PREFIX,3),name="sequence")
 if name in ["lstm","bilstm"]:
  layer=L.LSTM(64)
  x=(L.Bidirectional(layer) if name=="bilstm" else layer)(seq)
  x=L.Dropout(.3)(x);x=L.Dense(32,activation="relu")(x);inputs=seq
 else:
  ids=L.Input(shape=(2,),dtype="int32",name="endpoints")
  emb=L.Flatten()(KnownEmbedding(vocab_size)(ids))
  x=L.Concatenate()([L.Flatten()(seq),emb])
  x=L.Dense(64,activation="relu")(x);x=L.Dropout(.3)(x)
  inputs={"sequence":seq,"endpoints":ids}
 model=tf.keras.Model(inputs,L.Dense(6,activation="softmax")(x))
 model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
 return model

def inputs(d,name,identity=True):
 if name!="ip_embedding":return d["seq"]
 return {"sequence":d["seq"],"endpoints":d["endpoints"] if identity else np.zeros_like(d["endpoints"])}

def probabilities(model,x):
 n=len(next(iter(x.values()))) if isinstance(x,dict) else len(x)
 chunks=[]
 for i in range(0,n,256):
  part={k:v[i:i+256] for k,v in x.items()} if isinstance(x,dict) else x[i:i+256]
  chunks.append(model(part,training=False).numpy())
 return np.concatenate(chunks)

def fit_neural(name,tr,idx,vocab_size,identity):
 tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(SEED)
 model=build_model(name,vocab_size)
 x=inputs(tr,name,identity)
 x={k:v[idx] for k,v in x.items()} if isinstance(x,dict) else x[idx]
 ds=tf.data.Dataset.from_tensor_slices((x,tr["y"][idx]))
 ds=ds.shuffle(len(idx),seed=SEED,reshuffle_each_iteration=True).batch(32)
 opts=tf.data.Options();opts.experimental_deterministic=True;opts.threading.private_threadpool_size=1
 ds=ds.with_options(opts)
 begin=time.perf_counter();history=model.fit(ds,epochs=20,verbose=0)
 return model,history.history,time.perf_counter()-begin

def metrics(y,pred,labels):
 detail=classification_report(y,pred,labels=list(range(6)),target_names=labels,output_dict=True,zero_division=0)
 return {"accuracy":float(accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,labels=list(range(6)),average="macro",zero_division=0)),
 "balanced_accuracy":float(np.mean([detail[c]["recall"] for c in labels])),"per_class":{c:detail[c] for c in labels}}

def self_check():
 X=np.zeros((2,30,3));X[0,20,2]=1;X[1,0,2]=1
 assert active_mask(X).tolist()==[False,True]
 y=np.array([0,0,0,1]);idx=sample_indices(y,True)
 assert np.bincount(y[idx]).tolist()==[3,3]
 np.testing.assert_array_equal(idx,sample_indices(y,True))
 # Synthetic six-class fixture also checks train-only scaling and unknown endpoints.
 labels=np.repeat(np.array(["a","b","c","d","e","f"]),2)
 train={"X":np.ones((12,30,3)),"endpoints":np.full((12,2),"known"),"y":labels,
  "group":np.array(["g"+str(i) for i in range(12)])}
 valid={"X":np.full((12,30,3),100.),"endpoints":np.full((12,2),"unseen"),"y":labels,
  "group":np.array(["v"+str(i) for i in range(12)])}
 prepared,state,_=prepare({"train":train,"validation":valid})
 np.testing.assert_allclose(state["scaler"].mean_,np.full(3,np.log(2)))
 assert (prepared["validation"]["endpoints"]==0).all()
 print("Checks passed: prefix eligibility, resampling, training-only scaling, unknown endpoints.")

def run(feature_root,out):
 self_check()
 data,checks=load_data(feature_root);prep,state,coverage=prepare(data)
 environment={"python":sys.version,"platform":platform.platform(),"tensorflow":tf.__version__,
  "numpy":np.__version__,"pandas":pd.__version__,"sklearn":__import__("sklearn").__version__,
  "gpu_details":[str(tf.config.experimental.get_device_details(g)) for g in tf.config.list_physical_devices("GPU")]}
 signature=hashlib.sha256(json.dumps({"files":checks,"settings":SETTINGS,"environment":environment},sort_keys=True).encode()).hexdigest()
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 marker=out/"run_signature.txt"
 if marker.exists() and marker.read_text()!=signature:raise ValueError("Environment or inputs changed; use a new result folder")
 marker.write_text(signature);joblib.dump(state,out/"preprocessing.joblib")
 (out/"environment.json").write_text(json.dumps(environment,indent=2))
 tr,va=prep["train"],prep["validation"];labels=state["labels"];rows=[];errors=[]
 configs=[(mode,name,True) for mode in ["unbalanced","balanced"] for name in NAMES]
 configs += [(mode,"ip_embedding",False) for mode in ["unbalanced","balanced"]]
 for mode,name,identity in configs:
  tag=mode+"_"+name+("" if identity else "_no_identity");folder=out/tag;folder.mkdir(exist_ok=True)
  resultpath=folder/"metrics.json"
  if resultpath.exists():
   cached=json.loads(resultpath.read_text())
   if cached.get("run_signature")==signature and all((folder/p).exists() and digest(folder/p)==h for p,h in cached.get("artifact_hashes",{}).items()) and cached.get("artifact_hashes"):
    rows.append(cached);print("Reused",tag,flush=True);continue
  print("Training",tag,flush=True)
  try:
   idx=sample_indices(tr["y"],mode=="balanced");np.save(folder/"training_indices.npy",idx)
   classical=name in NAMES[:3];repeat=None
   if classical:
    model={"random_forest":lambda:RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=1),
     "svm":lambda:SVC(C=1.,gamma="scale"),"knn":lambda:KNeighborsClassifier(n_neighbors=5,n_jobs=1)}[name]()
    begin=time.perf_counter();model.fit(tr["flat"][idx],tr["y"][idx]);seconds=time.perf_counter()-begin
    pred=model.predict(va["flat"]);modelpath=folder/"model.joblib";joblib.dump(model,modelpath)
   else:
    model,history,seconds=fit_neural(name,tr,idx,len(state["endpoint_vocab"])+1,identity)
    vx=inputs(va,name,identity);probs=probabilities(model,vx);weights=[w.copy() for w in model.get_weights()]
    modelpath=folder/"model.keras";model.save(modelpath)
    (folder/"history.json").write_text(json.dumps(history,indent=2))
    del model
    repeated,_,_=fit_neural(name,tr,idx,len(state["endpoint_vocab"])+1,identity)
    p2=probabilities(repeated,vx);w2=repeated.get_weights()
    repeat={"weights_exact":len(weights)==len(w2) and all(np.array_equal(a,b) for a,b in zip(weights,w2)),
     "probabilities_exact":bool(np.array_equal(probs,p2)),"max_probability_difference":float(np.max(np.abs(probs-p2)))}
    (folder/"repeatability.json").write_text(json.dumps(repeat,indent=2))
    if not repeat["weights_exact"] or not repeat["probabilities_exact"]:
     raise RuntimeError("Neural repeatability failed; diagnostics saved")
    del repeated
    loaded=tf.keras.models.load_model(modelpath,compile=False)
    if not np.array_equal(probabilities(loaded,vx),probs):raise RuntimeError("Saved-model round-trip prediction mismatch")
    del loaded
    pred=probs.argmax(axis=1);np.save(folder/"validation_probabilities.npy",probs)
   row={"configuration":tag,"model":name,"balanced":mode=="balanced","endpoint_identity":name=="ip_embedding" and identity,
    **metrics(va["y"],pred,labels),"training_seconds_first_fit":seconds,"repeatability":repeat,"run_signature":signature}
   all_pred=np.full(len(data["validation"]["y"]),"NO_OBSERVED_TRAFFIC",dtype="U32")
   all_pred[va["mask"]]=np.array(labels)[pred]
   df=pd.DataFrame({"sample_id":data["validation"]["ids"],"capture":data["validation"]["capture"],
    "group":data["validation"]["group"],"true_label":data["validation"]["y"],
    "observed_traffic":va["mask"],"prediction":all_pred})
   df.to_csv(folder/"validation_predictions.csv",index=False)
   np.save(folder/"confusion.npy",confusion_matrix(va["y"],pred,labels=list(range(6))))
   groupmetrics=[]
   eligible=df[df.observed_traffic]
   for group,g in eligible.groupby("group"):
    groupmetrics.append({"group":group,"windows":len(g),"accuracy":float((g.true_label==g.prediction).mean())})
   (folder/"group_accuracy.json").write_text(json.dumps(groupmetrics,indent=2))
   row["artifact_hashes"]={p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name!="metrics.json"}
   resultpath.write_text(json.dumps(row,indent=2));rows.append(row)
  except Exception as error:
   errors.append({"configuration":tag,"type":type(error).__name__,"message":str(error)[:500]})
   print("Failed",tag,str(error),flush=True)
  (out/"errors.json").write_text(json.dumps(errors,indent=2))
 majority=int(np.bincount(tr["y"]).argmax())
 report={"stage":"gavist_validation_10seconds","settings":SETTINGS,"split_signature":SPLIT,"coverage":coverage,
  "completed_configurations":len(rows),"expected_configurations":14,"errors":errors,"results":rows,
  "majority_baseline_active_validation_accuracy":float(np.mean(va["y"]==majority)),
  "test_features_loaded":False,"test_evaluated":False,"publication_ready":False,
  "limitations":["Conditional metrics exclude prefixes without records; coverage reported separately.",
   "Single seed; neural duplicate fits check implementation repeatability, not scientific replication.",
   "Windows are correlated within groups; no confidence intervals or significance claim.",
   "Class balancing duplicates training windows and does not add independent data.",
   "IP embedding receives extra identity information; its no-identity condition is separate.",
   "Training times exclude the duplicate verification fit; no hardware-neutral speed comparison.",
   "This is ongoing-traffic classification, not session-start classification or measured QoS control."]}
 (out/"validation_summary.json").write_text(json.dumps(report,indent=2))
 table=pd.DataFrame([{k:r[k] for k in ["configuration","accuracy","macro_f1","balanced_accuracy"]} for r in rows])
 table.to_csv(out/"validation_comparison.csv",index=False)
 if len(table):
  import matplotlib.pyplot as plt
  ax=table.set_index("configuration")[["accuracy","macro_f1","balanced_accuracy"]].plot.bar(figsize=(13,6),ylim=(0,1))
  ax.set_title("10-second validation: active prefixes, fixed seed")
  fig=ax.get_figure();fig.tight_layout();fig.savefig(out/"validation_comparison.png",dpi=200);fig.savefig(out/"validation_comparison.pdf");plt.close(fig)
 return report
