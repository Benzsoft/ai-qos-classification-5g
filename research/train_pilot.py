"""Fixed-setting, single-seed six-model pilot. Never tunes against test data."""
from pathlib import Path
import hashlib,json,time,platform,sys,subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,classification_report,confusion_matrix
import joblib

SETTINGS={'seed':42,'prefix_packets':30,'epochs':20,'batch_size':32,'class_weight':None,
          'rf_trees':100,'svm_C':1.0,'svm_gamma':'scale','knn_k':5,
          'recurrent_units':64,'dense_units':32,'dropout':0.3,'embedding_dim':8,
          'optimizer':'Adam','learning_rate':0.001,'validation':None}
NAMES=['random_forest','svm','knn','lstm','bilstm','ip_embedding']

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_features(root):
    root=Path(root);summary=json.loads((root/'feature_summary.json').read_text())
    if summary.get('errors') or summary.get('completed_captures')!=24:raise ValueError('Incomplete feature extraction')
    manifest=json.loads((root/'source_manifest.json').read_text())
    if hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()!=summary['manifest_sha256']:raise ValueError('Manifest mismatch')
    packets={s:{k:[] for k in ['X','protocols','endpoints','sample_ids','y','capture','group']} for s in ['train','test']}
    checksums={};seen=set();groups={s:set() for s in packets}
    report_by_id={r['capture_id']:r for r in summary['reports']}
    for row in manifest:
        p=root/'captures'/(row['capture_id']+'.npz');sha=digest(p)
        if sha!=report_by_id[row['capture_id']]['npz_sha256']:raise ValueError('Feature checksum mismatch')
        checksums[row['capture_id']]=sha
        with np.load(p,allow_pickle=False) as z:
            n=len(z['X']);d=packets[row['split']]
            if z['X'].shape!=(n,30,3) or z['protocols'].shape!=(n,30) or z['endpoints'].shape!=(n,2):raise ValueError('Unexpected feature shape')
            if not np.isfinite(z['X']).all():raise ValueError('Nonfinite features')
            for sid in z['sample_ids'].tolist():
                if sid in seen:raise ValueError('Duplicate sample ID')
                seen.add(sid)
            for k in ['X','protocols','endpoints','sample_ids']:d[k].append(z[k].copy())
            for k,value in [('y',row['application']),('capture',row['capture_id']),('group',row['recording_group'])]:d[k].append(np.full(n,value))
            groups[row['split']].add(row['recording_group'])
    if groups['train']&groups['test']:raise ValueError('Recording-group leakage')
    data={s:{k:np.concatenate(v) for k,v in d.items()} for s,d in packets.items()}
    if any(len(d['X'])==0 for d in data.values()):raise ValueError('Empty partition')
    return data,checksums,summary

def preprocess(data):
    train=data['train'];labels=sorted(set(train['y']));labelmap={v:i for i,v in enumerate(labels)}
    if set(data['test']['y'])-set(labels):raise ValueError('Unknown test target')
    protocolmap={v:i+1 for i,v in enumerate(sorted(set(train['protocols'].ravel())))}
    endpointmap={v:i+1 for i,v in enumerate(sorted(set(train['endpoints'].ravel())))}
    # Global channel scales fitted ONLY to training packet prefixes.
    scaler=StandardScaler().fit(train['X'].reshape(-1,3))
    result={}
    for split,d in data.items():
        numerical=scaler.transform(d['X'].reshape(-1,3)).reshape(d['X'].shape).astype('float32')
        codes=np.array([protocolmap.get(v,0) for v in d['protocols'].ravel()],np.int32).reshape(d['protocols'].shape)
        onehot=np.eye(len(protocolmap)+1,dtype=np.float32)[codes]
        onehot[codes==0]=0  # Unseen protocols use zero vector, not an untrained active channel.
        seq=np.concatenate([numerical,onehot],axis=2)
        endpoints=np.array([endpointmap.get(v,0) for v in d['endpoints'].ravel()],np.int32).reshape(-1,2)
        result[split]={'seq':seq,'flat':seq.reshape(len(seq),-1),'endpoints':endpoints,
                       'y':np.array([labelmap[v] for v in d['y']],np.int32)}
    state={'labels':labels,'protocolmap':protocolmap,'endpointmap':endpointmap,'scaler':scaler}
    return result,state

def make_deep(name,shape,endpoint_count):
    import tensorflow as tf
    L=tf.keras.layers
    seq=L.Input(shape=shape,name='sequence')
    if name in ['lstm','bilstm']:
        recurrent=L.LSTM(64)
        x=(L.Bidirectional(recurrent) if name=='bilstm' else recurrent)(seq)
        x=L.Dropout(.3)(x);x=L.Dense(32,activation='relu')(x);inputs=seq
    else:
        ids=L.Input(shape=(2,),dtype='int32',name='endpoints')
        # Unknown index 0 has a fixed zero vector, avoiding an untrained random embedding.
        known=LambdaKnownEmbedding(endpoint_count,8)
        emb=known(ids)
        x=L.Concatenate()([L.Flatten()(seq),L.Flatten()(emb)])
        x=L.Dense(64,activation='relu')(x);x=L.Dropout(.3)(x);inputs={'sequence':seq,'endpoints':ids}
    output=L.Dense(3,activation='softmax')(x)
    model=tf.keras.Model(inputs,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

# Defined lazily to keep data/preprocessing checks usable without TensorFlow.
def LambdaKnownEmbedding(count,width):
    import tensorflow as tf
    @tf.keras.utils.register_keras_serializable(package='qos')
    class KnownEmbedding(tf.keras.layers.Layer):
        def __init__(self,count=count,width=width,**kwargs):
            super().__init__(**kwargs);self.count=count;self.width=width
            self.embedding=tf.keras.layers.Embedding(count,width)
        def call(self,ids):
            return self.embedding(ids)*tf.cast(tf.expand_dims(ids!=0,-1),tf.float32)
        def get_config(self):return {**super().get_config(),'count':self.count,'width':self.width}
    return KnownEmbedding()

def run_pilot(feature_root,out):
    import tensorflow as tf
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data,checksums,feature_summary=load_features(feature_root);prepared,state=preprocess(data)
    if len(state['labels'])!=3:raise ValueError('This fixed pilot expects three classes')
    tf.keras.utils.set_random_seed(42)
    for gpu in tf.config.list_physical_devices('GPU'):
        try:tf.config.experimental.set_memory_growth(gpu,True)
        except RuntimeError:pass
    environment={'python':sys.version,'platform':platform.platform(),'tensorflow':tf.__version__,
                 'numpy':np.__version__,'pandas':pd.__version__,'sklearn':__import__('sklearn').__version__}
    try:environment['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    except FileNotFoundError:environment['gpu']='none'
    signature=hashlib.sha256(json.dumps({'files':checksums,'settings':SETTINGS,'environment':environment,'implementation_version':1},sort_keys=True).encode()).hexdigest()
    marker=out/'run_signature.txt'
    if marker.exists() and marker.read_text()!=signature:raise ValueError('Inputs/environment changed: use a new output folder')
    marker.write_text(signature)
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    (out/'packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (out/'settings.json').write_text(json.dumps(SETTINGS,indent=2))
    joblib.dump(state,out/'preprocessing.joblib')
    tr,te=prepared['train'],prepared['test'];metrics=[];errors=[]
    for name in NAMES:
        result_path=out/(name+'_metrics.json')
        model_file=out/(name+('.joblib' if name in NAMES[:3] else '.keras'))
        if result_path.exists() and model_file.exists() and (out/(name+'_predictions.csv')).exists() and (out/(name+'_confusion.npy')).exists():
            metrics.append(json.loads(result_path.read_text()));print('Reusing completed:',name,flush=True);continue
        try:
            tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)
            print('Training:',name,flush=True)
            classical=name in NAMES[:3]
            if classical:
                model={'random_forest':lambda:RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=1),
                       'svm':lambda:SVC(C=1.,gamma='scale',kernel='rbf'),
                       'knn':lambda:KNeighborsClassifier(n_neighbors=5,n_jobs=1)}[name]()
                start=time.perf_counter();model.fit(tr['flat'],tr['y']);seconds=time.perf_counter()-start
                predict=lambda x:model.predict(x)
                model_path=out/(name+'.joblib');joblib.dump(model,model_path)
                test_input=te['flat'];truth=te['y']
            else:
                model=make_deep(name,tr['seq'].shape[1:],len(state['endpointmap'])+1)
                train_input={'sequence':tr['seq'],'endpoints':tr['endpoints']} if name=='ip_embedding' else tr['seq']
                test_input={'sequence':te['seq'],'endpoints':te['endpoints']} if name=='ip_embedding' else te['seq']
                ds=tf.data.Dataset.from_tensor_slices((train_input,tr['y'])).shuffle(len(tr['y']),seed=42,reshuffle_each_iteration=True).batch(32)
                options=tf.data.Options();options.threading.private_threadpool_size=1;ds=ds.with_options(options)
                start=time.perf_counter();history=model.fit(ds,epochs=20,verbose=2);seconds=time.perf_counter()-start
                (out/(name+'_history.json')).write_text(json.dumps(history.history))
                model_path=out/(name+'.keras');model.save(model_path)
                predict=lambda x:np.argmax(model(x,training=False).numpy(),axis=1)
                truth=te['y']
            pred=predict(test_input)
            # Warmed in-process batch-one latency, excluding preprocessing and observation.
            times=[]
            for i in range(min(100,len(truth))+5):
                j=i%len(truth)
                one={k:v[j:j+1] for k,v in test_input.items()} if isinstance(test_input,dict) else test_input[j:j+1]
                start=time.perf_counter();predict(one);elapsed=(time.perf_counter()-start)*1000
                if i>=5:times.append(elapsed)
            detail=classification_report(truth,pred,labels=[0,1,2],target_names=state['labels'],output_dict=True,zero_division=0)
            row={'model':name,'accuracy':float(accuracy_score(truth,pred)),
                 'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0)),
                 'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
                 'training_seconds':seconds,'inference_median_ms':float(np.median(times)),
                 'inference_p95_ms':float(np.percentile(times,95)), 'model_bytes':model_path.stat().st_size,
                 'execution':'CPU sklearn' if classical else ('TensorFlow GPU available' if tf.config.list_physical_devices('GPU') else 'TensorFlow CPU'),
                 'endpoint_identity_features':name=='ip_embedding','classification_report':detail}
            cm=confusion_matrix(truth,pred,labels=[0,1,2]);np.save(out/(name+'_confusion.npy'),cm,allow_pickle=False)
            pd.DataFrame({'sample_id':data['test']['sample_ids'],'capture_id':data['test']['capture'],
                          'recording_group':data['test']['group'],'true_label':data['test']['y'],
                          'predicted_label':np.array(state['labels'])[pred]}).to_csv(out/(name+'_predictions.csv'),index=False)
            result_path.write_text(json.dumps(row,indent=2));metrics.append(row)
        except Exception as error:
            errors.append({'model':name,'type':type(error).__name__,'message':str(error)[:500]});print('Model failed:',name,str(error),flush=True)
    counts={s:{c:int(np.sum(data[s]['y']==c)) for c in state['labels']} for s in data}
    summary={'stage':'fixed_settings_30_packet_pilot','settings':SETTINGS,'counts':counts,'models_completed':len(metrics),
             'errors':errors,'results':metrics,'environment':environment,'run_signature':signature,
             'majority_baseline_accuracy':float(np.mean(te['y']==np.bincount(tr['y']).argmax())),
             'test_unknown_endpoint_fraction':float(np.mean(te['endpoints']==0)),
             'publication_ready':False,'limitations':['Single seed; no confidence intervals or significance claim.',
             'Only 16 YouTube training segments; severe imbalance, no weighting or resampling in this baseline.',
             'IP embedding receives extra endpoint identity; comparison is not identical feature information.',
             'Timing mixes CPU classical and TensorFlow execution; not a hardware-neutral speed ranking.',
             '30-packet endpoint segments only; not true flow starts or QoS control results.',
             'Raw protocol dissector labels may encode capture-specific artifacts; ablation required.']}
    (out/'pilot_summary.json').write_text(json.dumps(summary,indent=2))
    table=pd.DataFrame([{k:v for k,v in r.items() if k!='classification_report'} for r in metrics])
    table.to_csv(out/'model_comparison.csv',index=False)
    perclass=[]
    for r in metrics:
        for c in state['labels']:perclass.append({'model':r['model'],'class':c,**r['classification_report'][c]})
    pd.DataFrame(perclass).to_csv(out/'per_class_metrics.csv',index=False)
    import matplotlib.pyplot as plt
    if len(table):
        fig,ax=plt.subplots(figsize=(10,4));table.set_index('model')[['accuracy','macro_f1','balanced_accuracy']].plot.bar(ax=ax)
        ax.set_ylim(0,1);ax.set_title('Preliminary pilot — fixed settings, single seed');fig.tight_layout()
        fig.savefig(out/'comparison.png',dpi=200);fig.savefig(out/'comparison.pdf');plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for ax,name in zip(axes.flat,NAMES):
            p=out/(name+'_confusion.npy')
            if not p.exists():ax.set_axis_off();continue
            cm=np.load(p);ax.imshow(cm,cmap='Blues');ax.set_title(name)
            ax.set_xticks(range(3),state['labels'],rotation=35,ha='right');ax.set_yticks(range(3),state['labels']);ax.set_xlabel('Predicted');ax.set_ylabel('True')
            for i in range(3):
                for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        fig.tight_layout();fig.savefig(out/'confusion_matrices.png',dpi=200);fig.savefig(out/'confusion_matrices.pdf');plt.close(fig)
    return summary,table
