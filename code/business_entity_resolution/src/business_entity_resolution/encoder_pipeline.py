"""Durable encoder-first ER pipeline. No external entity data is used.

Selection uses calibration entities; an untouched holdout is evaluated once.
Every long stage is resumable and the entrypoint is suitable for nohup.
"""
from __future__ import annotations
import argparse, fcntl, gc, hashlib, json, os, resource, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
import torch
import xgboost as xgb
from .config import DEFAULT_CONFIG, REPO_ROOT
from .split import stable_fraction
from .stress_test import _load_truth
from .evaluation import evaluate_predictions, candidate_metrics, evaluate_breakdowns

ROOT=REPO_ROOT
WORK=ROOT/'cache/encoder_native_v1'
LOG=ROOT/'logs'
MODEL='intfloat/multilingual-e5-base'
VERSION='encoder_native_v1'
SEED=260926
START=time.time()
STATE={}


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def save_frame(frame,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.parquet.tmp'); frame.to_parquet(tmp,index=False); tmp.replace(path)


def event(kind,**data):
    record={'event':kind,'time':time.strftime('%Y-%m-%d %H:%M:%S'),'elapsed_seconds':time.time()-START,**data}
    print(json.dumps(record,default=str),flush=True)
    LOG.mkdir(exist_ok=True)
    with (LOG/'encoder_experiments.jsonl').open('a') as f:f.write(json.dumps(record,default=str)+'\n')


def status(stage,**kwargs):
    STATE.pop('reached_0.98_clean_holdout',None)
    STATE.setdefault('started_at_epoch',START)
    STATE.update(kwargs); STATE['stage']=stage; STATE['pid']=os.getpid(); STATE['updated']=time.strftime('%Y-%m-%d %H:%M:%S')
    STATE.setdefault('latest_clean_holdout_score','not evaluated'); STATE.setdefault('best_calibration_score','pending')
    STATE.setdefault('reached_0_98_clean_holdout',False); STATE.setdefault('validator','NOT RUN'); STATE.setdefault('id_check','NOT RUN')
    STATE.setdefault('failures',[]); STATE.setdefault('output_paths',['output/matching_results.tsv','output/candidate_pairs.tsv'])
    STATE['baseline']=.965614; STATE['target']=.98
    score=STATE.get('latest_clean_holdout_score')
    STATE['beat_baseline']=isinstance(score,(int,float)) and score>.965614
    STATE.setdefault('submission_archive','not created')
    STATE.setdefault('current_bottleneck','full-test encoding and inference')
    STATE['log']='logs/encoder_pipeline.log'; STATE['state_path']=str(WORK/'state.json')
    atomic_json(WORK/'state.json',STATE)
    tmp=LOG/'ASTRA_STATUS.md.tmp'
    tmp.write_text('# Encoder-first submission status\n\n'+ '\n'.join(f'- **{k}**: {v}' for k,v in STATE.items())+'\n')
    tmp.replace(LOG/'ASTRA_STATUS.md')


def hardware():
    cpus=len(os.sched_getaffinity(0)); slurm=os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm:cpus=min(cpus,int(slurm))
    torch.set_num_threads(min(8,cpus)); torch.set_num_interop_threads(2)
    torch.manual_seed(SEED); np.random.seed(SEED)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; launch outside sandbox with V100 access')
    event('hardware',cpu_count=os.cpu_count(),psutil_count=psutil.cpu_count(),affinity=sorted(os.sched_getaffinity(0)),threads=min(8,cpus),gpu=torch.cuda.get_device_name(0),ram_available=psutil.virtual_memory().available,thread_env={k:os.environ.get(k) for k in ['SLURM_CPUS_PER_TASK','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
    return cpus


def texts(frame,role='query'):
    return (role+': business name: '+frame.name_full.fillna('').astype(str).str.strip()+' | address: '+frame.address_full.fillna('').astype(str).str.strip()+' | country: '+frame.country_norm.fillna('').astype(str).str.strip()).tolist()


def snapshot():
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(MODEL,revision='d128750597153bb5987e10b1c3493a34e5a4502a',local_files_only=True)
    except Exception:
        return snapshot_download(MODEL,revision='d128750597153bb5987e10b1c3493a34e5a4502a')


def fingerprint_ids(frame):
    return hashlib.sha256('\n'.join(frame.entity_id).encode()).hexdigest()


def encode(frame,role,path,model,batch=512):
    path=Path(path); marker=path.with_suffix('.json')
    identity={'ids_sha256':fingerprint_ids(frame),'shape':[len(frame),768],'model':MODEL,'revision':Path(snapshot()).name,'max_length':128,'role':role,'dtype':'float16','precision':'autocast_fp16'}
    done=json.load(marker.open()) if marker.exists() else {}
    if done.get('identity')==identity and path.exists() and done.get('complete'):
        arr=np.load(path,mmap_mode='r'); assert list(arr.shape)==identity['shape']; event('embedding_reused',path=str(path),rows=len(frame));return arr
    path.parent.mkdir(parents=True,exist_ok=True); partial=path.with_suffix('.partial.npy')
    offset=done.get('rows',0) if done.get('identity')==identity and partial.exists() else 0
    arr=np.lib.format.open_memmap(partial,mode='r+' if offset else 'w+',dtype=np.float16,shape=(len(frame),768))
    started=time.time()
    for start in range(offset,len(frame),8192):
        end=min(start+8192,len(frame)); data=texts(frame.iloc[start:end],role)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            result=model.encode(data,batch_size=batch,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite embeddings')
        arr[start:end]=result.astype(np.float16);arr.flush()
        atomic_json(marker,{'identity':identity,'rows':end,'complete':False})
        event('embedding_progress',path=str(path),rows=end,total=len(frame),rows_per_sec=(end-offset)/max(.01,time.time()-started),vram=torch.cuda.max_memory_allocated()/2**30)
        status('encoding',current_embedding=str(path),embedding_rows=f'{end}/{len(frame)}')
    del arr;partial.replace(path);atomic_json(marker,{'identity':identity,'rows':len(frame),'complete':True,'seconds':time.time()-started})
    return np.load(path,mmap_mode='r')


def encode_benchmark(model,frame):
    sample=texts(frame.iloc[:2048]);results=[]
    for batch in [256,512,1024,2048]:
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.time()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                out=model.encode(sample,batch_size=batch,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            torch.cuda.synchronize();seconds=time.time()-started
            assert np.isfinite(out).all();results.append((seconds,batch));event('embedding_batch_benchmark',batch=batch,seconds=seconds,rows_per_sec=len(sample)/seconds,peak_vram=torch.cuda.max_memory_allocated()/2**30)
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache();event('embedding_batch_oom',batch=batch)
    return min(results)[1]


def gpu_retrieve(q,f,qcountry,fcountry,k=40):
    """Blocked exhaustive cosine over float16 stored vectors; no full universe matrix."""
    scores=np.full((len(q),k),-np.inf,np.float32);indices=np.full((len(q),k),-1,np.int32)
    for country in sorted(set(qcountry)):
        qi=np.flatnonzero(qcountry==country);fi=np.flatnonzero(fcountry==country)
        if len(fi)==0:continue
        # Up to ~8 GiB for the largest feed shard; no repeated host copies.
        feed=torch.as_tensor(np.array(f[fi],copy=True),device='cuda',dtype=torch.float16)
        for start in range(0,len(qi),256):
            qs=qi[start:start+256];query=torch.as_tensor(np.array(q[qs],copy=True),device='cuda',dtype=torch.float16)
            best=torch.full((len(qs),k),-torch.inf,device='cuda');bestidx=torch.full((len(qs),k),-1,device='cuda',dtype=torch.long)
            for fs in range(0,len(fi),100000):
                sims=(query@feed[fs:fs+100000].T).float()
                ss,ii=torch.topk(sims,min(k,sims.shape[1]),dim=1);ii+=fs
                joined=torch.cat((best,ss),1);ids=torch.cat((bestidx,ii),1)
                best,pos=torch.topk(joined,k,dim=1);bestidx=ids.gather(1,pos)
            scores[qs]=best.cpu().numpy();local=bestidx.cpu().numpy();indices[qs]=np.where(local>=0,fi[np.maximum(local,0)],-1)
        del feed;torch.cuda.empty_cache()
    return scores,indices


class StructuralIndex:
    """Bounded exact keys and rare-address-token postings. No TF-IDF pass."""
    def __init__(self,feed):
        self.feed=feed
        self.keys={}
        for field in ['name_full','name_core','address_full']:
            key=feed.country_norm+'\x1f'+feed[field]
            codes,unique=pd.factorize(key,sort=False)
            order=np.argsort(codes,kind='stable').astype(np.int32)
            counts=np.bincount(codes);offsets=np.r_[0,np.cumsum(counts)]
            self.keys[field]={str(value):order[offsets[i]:offsets[i+1]] for i,value in enumerate(unique) if counts[i]<=100 and str(value).split('\x1f',1)[1]}
            event('structural_index_field',field=field,keys=len(self.keys[field]),feed_rows=len(feed))
        self.cv=CountVectorizer(binary=True,lowercase=False,token_pattern=r'(?u)\b\w{3,}\b',max_df=500,dtype=np.float32)
        try:
            self.address=self.cv.fit_transform(feed.address_tokens).tocsr();self.address_t=self.address.T.tocsr()
        except ValueError:
            self.address=None
        self.length=feed.address_tokens.str.split().str.len().to_numpy()

    def retrieve(self,source1,topk=20):
        rows=[]
        for field,mapping in self.keys.items():
            for i,key in enumerate((source1.country_norm+'\x1f'+source1[field]).to_numpy()):
                ids=mapping.get(key)
                if ids is not None:rows.extend((i,int(j),field) for j in ids)
        if self.address is not None:
            query=self.cv.transform(source1.address_tokens)
            for start in range(0,len(source1),256):
                shared=(query[start:start+256]@self.address_t).tocsr()
                for local in range(shared.shape[0]):
                    a,b=shared.indptr[local:local+2];js=shared.indices[a:b];values=shared.data[a:b]
                    country=source1.country_norm.iloc[start+local];mask=self.feed.country_norm.to_numpy()[js]==country;js=js[mask];values=values[mask]
                    if len(js):
                        values=values/(1+self.length[js]);order=np.lexsort((js,-values))[:topk]
                        rows.extend((start+local,int(j),'rare_address') for j in js[order])
        return rows


def candidates(source1,feed,q,f,k=40,index=None):
    started=time.time();ss,jj=gpu_retrieve(q,f,source1.country_norm.to_numpy(),feed.country_norm.to_numpy(),k)
    ii=np.repeat(np.arange(len(source1)),k);js=jj.ravel();valid=js>=0
    out=pd.DataFrame({'qi':ii[valid],'fi':js[valid],'e5_rank':np.tile(np.arange(1,k+1),len(source1))[valid],'e5_similarity':ss.ravel()[valid]})
    out['semantic']=1
    structural=(index or StructuralIndex(feed)).retrieve(source1)
    if structural:
        extra=pd.DataFrame(structural,columns=['qi','fi','channel'])
        prov=pd.crosstab([extra.qi,extra.fi],extra.channel).clip(upper=1).reset_index()
        out=out.merge(prov,on=['qi','fi'],how='outer',validate='one_to_one')
    for col in ['semantic','e5_rank','name_full','name_core','address_full','rare_address']:
        if col not in out:out[col]=0
        out[col]=out[col].fillna(0)
    missing=out.e5_similarity.isna().to_numpy()
    for start in range(0,int(missing.sum()),4096):
        positions=np.flatnonzero(missing)[start:start+4096];a=out.qi.to_numpy()[positions];b=out.fi.to_numpy()[positions]
        out.loc[positions,'e5_similarity']=np.einsum('ij,ij->i',np.asarray(q[a],np.float32),np.asarray(f[b],np.float32))
    out['source1_entity_id']=source1.entity_id.to_numpy()[out.qi.to_numpy()]
    out['candidate_entity_id']=feed.entity_id.to_numpy()[out.fi.to_numpy()]
    out.sort_values(['qi','e5_similarity','candidate_entity_id'],ascending=[True,False,True],kind='stable',inplace=True)
    out.reset_index(drop=True,inplace=True)
    assert not out.duplicated(['qi','fi']).any()
    event('candidate_generation',seconds=time.time()-started,source1=len(source1),pairs=len(out))
    return out



def add_address_view(pairs,s1,feed,q,f,model=None,production_dir=None,qa=None,fa=None):
    """Independent address E5 view; validate its incremental value before deployment."""
    from sentence_transformers import SentenceTransformer
    if qa is None:
        own=model is None
        if own:model=SentenceTransformer(snapshot(),device='cuda');model.max_seq_length=128
        directory=Path(production_dir) if production_dir is not None else WORK/'address_view'
        sq=s1.copy();ff=feed.copy();sq['name_full']='';ff['name_full']=''
        qa=encode(sq,'query',directory/'source1.npy',model,512)
        fa=encode(ff,'passage',directory/'feed.npy',model,512)
        if own:del model;gc.collect();torch.cuda.empty_cache()
    score,idx=gpu_retrieve(qa,fa,s1.country_norm.to_numpy(),feed.country_norm.to_numpy(),20)
    ii=np.repeat(np.arange(len(s1)),20);jj=idx.ravel();valid=jj>=0
    address=pd.DataFrame({'qi':ii[valid],'fi':jj[valid],'address_e5_similarity':score.ravel()[valid],'address_e5_rank':np.tile(np.arange(1,21),len(s1))[valid]})
    merged=pairs.merge(address,on=['qi','fi'],how='outer',validate='one_to_one')
    merged['qi']=merged.qi.astype(np.int64);merged['fi']=merged.fi.astype(np.int64)
    for col in ['semantic','e5_rank','name_full','name_core','address_full','rare_address','address_e5_similarity','address_e5_rank']:merged[col]=merged[col].fillna(0)
    missing=merged.e5_similarity.isna().to_numpy();positions=np.flatnonzero(missing)
    for start in range(0,len(positions),4096):
        pp=positions[start:start+4096];a=merged.qi.to_numpy()[pp];b=merged.fi.to_numpy()[pp]
        merged.loc[pp,'e5_similarity']=np.einsum('ij,ij->i',np.asarray(q[a],np.float32),np.asarray(f[b],np.float32))
    merged['source1_entity_id']=s1.entity_id.to_numpy()[merged.qi.to_numpy()];merged['candidate_entity_id']=feed.entity_id.to_numpy()[merged.fi.to_numpy()]
    merged.sort_values(['qi','e5_similarity','candidate_entity_id'],ascending=[True,False,True],kind='stable',inplace=True)
    return merged.reset_index(drop=True)

def lexical_pair_stats(left,right,char=False):
    """Sparse vectorized per-pair overlap; there is no pair-universe multiplication."""
    joined=pd.concat([left,right],ignore_index=True)
    codes,unique=pd.factorize(joined,sort=False)
    if char:
        vectorizer=HashingVectorizer(analyzer='char',ngram_range=(2,3),n_features=2**18,alternate_sign=False,norm='l2',lowercase=False,dtype=np.float32)
        matrix=vectorizer.transform(unique)
        a=matrix[codes[:len(left)]];b=matrix[codes[len(left):]]
        return np.asarray(a.multiply(b).sum(axis=1)).ravel()
    cv=CountVectorizer(binary=True,lowercase=False,token_pattern=r'(?u)\b\w+\b',dtype=np.float32)
    try: matrix=cv.fit_transform(unique)
    except ValueError:return np.zeros(len(left)),np.ones(len(left)),np.zeros(len(left)),np.zeros(len(left))
    a=matrix[codes[:len(left)]];b=matrix[codes[len(left):]]
    shared=np.asarray(a.multiply(b).sum(axis=1)).ravel()
    na=np.diff(a.indptr);nb=np.diff(b.indptr);union=na+nb-shared
    return shared,np.divide(shared,union,out=np.ones(len(shared)),where=union>0),na,nb


_FEATURE_POOL=None


def _feature_worker_init():
    torch.set_num_threads(1)


def _pair_feature_chunk(payload):
    p,l,r=payload
    x={}
    for field,prefix in [('name_full','name'),('address_full','address')]:
        a=l[field];b=r[field];la=a.str.len().to_numpy();lb=b.str.len().to_numpy()
        x[prefix+'_exact']=((a==b)&a.ne('')).to_numpy();x[prefix+'_length_ratio']=np.divide(np.minimum(la,lb),np.maximum(la,lb),out=np.ones(len(a)),where=np.maximum(la,lb)>0)
        shared,jacc,na,nb=lexical_pair_stats(a,b)
        x[prefix+'_jaccard']=jacc;x[prefix+'_shared']=shared;x[prefix+'_token_difference']=abs(na-nb)
        x[prefix+'_overlap']=np.divide(shared,np.minimum(na,nb),out=np.zeros(len(a)),where=np.minimum(na,nb)>0)
        x[prefix+'_character_cosine']=lexical_pair_stats(a,b,char=True)
        x[prefix+'_prefix3']=(a.str[:3]==b.str[:3]).to_numpy();x[prefix+'_suffix3']=(a.str[-3:]==b.str[-3:]).to_numpy()
        x[prefix+'_missing_left']=a.eq('').to_numpy();x[prefix+'_missing_right']=b.eq('').to_numpy()
        an=a.str.findall(r'\d+').str.join(' ');bn=b.str.findall(r'\d+').str.join(' ')
        ns,nj,nl,nr=lexical_pair_stats(an,bn)
        x[prefix+'_numeric_overlap']=np.where((nl>0)&(nr>0),nj,0)
        x[prefix+'_numeric_conflict']=((nl>0)&(nr>0)&(ns==0))
        x[prefix+'_numeric_equal']=((an==bn)&an.ne('')).to_numpy()
        x[prefix+'_first_number_equal']=((an.str.split().str[0]==bn.str.split().str[0])&an.ne('')).to_numpy()
    x['name_core_exact']=(l.name_core.eq(r.name_core)&l.name_core.ne('')).to_numpy()
    if 'acronym' in l:x['acronym_equal']=(l.acronym.eq(r.acronym)&l.acronym.ne('')).to_numpy()
    x['postcode_equal']=(l.postcode.eq(r.postcode)&l.postcode.ne('')).to_numpy()
    x['postcode_conflict']=(l.postcode.ne(r.postcode)&l.postcode.ne('')&r.postcode.ne('')).to_numpy()
    x['same_country']=l.country_norm.eq(r.country_norm).to_numpy()
    x['source2']=r.entity_id.str.startswith('S2-').to_numpy()
    for col in ['e5_similarity','e5_rank','semantic','name_full','name_core','address_full','rare_address']:x['retrieval_'+col]=p[col].to_numpy()
    for col in ['address_e5_similarity','address_e5_rank']:
        if col in p:x[col]=p[col].to_numpy()
    return pd.DataFrame(x).astype(np.float32)


def build_features(pairs,s1,feed):
    global _FEATURE_POOL
    chunks=[];started=time.time()
    workers=min(int(os.environ.get('ENCODER_FEATURE_WORKERS','1')),len(os.sched_getaffinity(0)))
    if workers>1 and _FEATURE_POOL is None:
        _FEATURE_POOL=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_feature_worker_init)
    pending=[]
    for start in range(0,len(pairs),50000):
        p=pairs.iloc[start:start+50000]
        l=s1.iloc[p.qi.to_numpy()].reset_index(drop=True)
        r=feed.iloc[p.fi.to_numpy()].reset_index(drop=True)
        if workers>1:
            pending.append(_FEATURE_POOL.submit(_pair_feature_chunk,(p,l,r)))
            if len(pending)>=workers:
                chunks.append(pending.pop(0).result())
        else:chunks.append(_pair_feature_chunk((p,l,r)))
    chunks.extend(future.result() for future in pending)
    result=pd.concat(chunks,ignore_index=True)
    scores=pairs.e5_similarity;group=scores.groupby(pairs.qi,sort=False)
    best=group.transform('max');result['e5_gap_best']=(best-scores).to_numpy();result['e5_gap_next']=(scores-group.shift(-1).fillna(scores)).to_numpy()
    result['e5_reciprocal_rank']=1/(1+pairs.e5_rank.to_numpy());result['candidate_count']=group.transform('size').to_numpy()
    result['score_percentile']=group.rank(pct=True,method='average').to_numpy()
    result['score_std']=group.transform('std').fillna(0).to_numpy()
    for delta in [.005,.01,.02]:result[f'competitors_{delta}']=(scores>=best-delta).groupby(pairs.qi).transform('sum').to_numpy()
    assert np.isfinite(result.to_numpy()).all()
    event('features',pairs=len(pairs),feature_count=len(result.columns),seconds=time.time()-started)
    return result.astype(np.float32)


def labeled(pairs,truth):
    target=pd.MultiIndex.from_tuples([(s,t) for s,ts in truth.items() for t in ts])
    return pd.MultiIndex.from_frame(pairs[['source1_entity_id','candidate_entity_id']]).isin(target).astype(np.uint8)


def metric_arrays(qindex,truthcounts,label,selected,entities):
    counts=np.bincount(qindex[selected],minlength=len(truthcounts));tp=np.bincount(qindex[selected],weights=label[selected],minlength=len(truthcounts))
    denom=.25*truthcounts+counts
    f=np.divide(1.25*tp,denom,out=np.zeros(len(tp)),where=denom>0);f[(truthcounts==0)&(counts==0)]=1
    return float(f[entities].mean())


def tune(prob,pairs,y,truthcounts,entitymask,countries):
    qi=pairs.qi.to_numpy();selected_rows=entitymask[qi]
    thresholds=np.unique(np.r_[np.linspace(.05,.99,120),np.linspace(.99,.9999,20)])
    def best(mask):
        scored=[(metric_arrays(qi,truthcounts,y,(prob>=t)&selected_rows,mask),float(t)) for t in thresholds]
        return max(scored)[1]
    global_t=best(entitymask);config={'global':global_t,'countries':{}}
    for country in sorted(set(countries)):
        mask=entitymask&(countries==country)
        if mask.sum()>=100:config['countries'][country]=best(mask)
    global_score=metric_arrays(qi,truthcounts,y,(prob>=global_t)&selected_rows,entitymask)
    per=np.array([config['countries'].get(c,global_t) for c in countries])[qi]
    country_score=metric_arrays(qi,truthcounts,y,(prob>=per)&selected_rows,entitymask)
    if country_score<=global_score:config['countries']={}
    return config,max(global_score,country_score)


def decode(prob,pairs,s1,thresholds):
    ts=np.array([thresholds['countries'].get(c,thresholds['global']) for c in s1.country_norm])[pairs.qi.to_numpy()]
    return prob>=ts


def report(prob,pairs,s1,truth,mask,thresholds):
    selected=decode(prob,pairs,s1,thresholds)&mask[pairs.qi.to_numpy()]
    pred=pairs.loc[selected].groupby('source1_entity_id').candidate_entity_id.agg(set).to_dict()
    subset=s1.loc[mask];m=evaluate_predictions(subset.entity_id,truth,pred)
    m.update(evaluate_breakdowns(subset,truth,pred));m['FP']=m['predicted_pairs']-m['true_positive_pairs'];m['FN']=m['true_pairs']-m['true_positive_pairs']
    candidate=pairs.loc[mask[pairs.qi.to_numpy()]];m['retrieval']=candidate_metrics(subset.entity_id.tolist(),truth,candidate)
    m['thresholds']=thresholds
    return m


def train_xgb(features,y,training,calibration,output,rounds=1600):
    if output.exists():return xgb.Booster(model_file=output)
    print('🔥 MODEL TRAINING STARTS NOW: GPU XGBOOST ENCODER-AWARE CLASSIFIER',flush=True)
    params={'objective':'binary:logistic','tree_method':'hist','device':'cuda','max_depth':7,'eta':.04,'subsample':.85,'colsample_bytree':.9,'min_child_weight':8,'lambda':2,'seed':SEED,'nthread':8,'eval_metric':'logloss'}
    started=time.time();dm=xgb.DMatrix(features.loc[training],label=y[training]);dv=xgb.DMatrix(features.loc[calibration],label=y[calibration])
    model=xgb.train(params,dm,num_boost_round=rounds,evals=[(dv,'calibration')],early_stopping_rounds=80,verbose_eval=100)
    model.save_model(output);atomic_json(output.with_suffix('.metadata.json'),{'params':params,'best_iteration':model.best_iteration,'features':features.columns.tolist(),'train_positive':int(y[training].sum()),'train_negative':int(training.sum()-y[training].sum()),'seconds':time.time()-started})
    return model


def xgb_predict(model,features):
    model.set_param({'device':'cuda','nthread':8})
    return model.predict(xgb.DMatrix(features),iteration_range=(0,int(model.attr('best_iteration') or model.num_boosted_rounds()-1)+1))


def pair_texts(pairs,s1,feed):
    left=s1.iloc[pairs.qi.to_numpy()];right=feed.iloc[pairs.fi.to_numpy()]
    a=(left.name_full+' | '+left.address_full+' | '+left.country_norm).tolist()
    b=(right.name_full+' | '+right.address_full+' | '+right.country_norm).tolist()
    return a,b


def ce_predict(model,tokenizer,pairs,s1,feed,batch=128):
    if len(pairs)==0:return np.zeros(0,np.float32)
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(pairs),batch):
            a,b=pair_texts(pairs.iloc[start:start+batch],s1,feed)
            inputs=tokenizer(a,b,padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda')
            with torch.autocast('cuda',dtype=torch.float16):scores=model(**inputs).logits[:,0].float().sigmoid()
            out.append(scores.cpu().numpy())
    return np.concatenate(out)


def train_ce(pairs,s1,feed,y,trainmask,calmask,baseprob,truthcounts,countries):
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    directory=WORK/'cross_encoder';tokenizer=AutoTokenizer.from_pretrained(snapshot(),local_files_only=True)
    result_path=directory/'selection.json'
    if result_path.exists():return json.load(result_path.open())
    directory.mkdir(exist_ok=True)
    status('cross_encoder_pilot',cross_encoder=MODEL+' pair classifier',next_step='one epoch hard negatives; calibration selection')
    print('🔥 MODEL TRAINING STARTS NOW: intfloat/multilingual-e5-base CROSS-ENCODER',flush=True)
    model=AutoModelForSequenceClassification.from_pretrained(snapshot(),num_labels=1,local_files_only=True).to('cuda')
    # Train the last four layers and classification head; frozen lower layers
    # preserve multilingual representations while bounding pilot cost.
    for name,param in model.named_parameters():
        param.requires_grad=('classifier' in name or any(f'encoder.layer.{i}.' in name for i in range(8,12)))
    trainrows=trainmask[pairs.qi.to_numpy()];calrows=calmask[pairs.qi.to_numpy()]
    negative=pairs.loc[trainrows&(y==0),['qi']].copy();negative['hardness']=baseprob[trainrows&(y==0)]
    positives=np.flatnonzero(trainrows&(y==1));npositive=pd.Series(y[trainrows],index=pairs.qi.to_numpy()[trainrows]).groupby(level=0).sum()
    negative=negative.sort_values(['qi','hardness'],ascending=[True,False]);rank=negative.groupby('qi').cumcount()
    selectedneg=negative.index[rank<negative.qi.map(npositive).clip(lower=1)*5].to_numpy()
    rows=np.r_[positives,selectedneg];rng=np.random.default_rng(SEED);rng.shuffle(rows)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');batch=32;steps=int(np.ceil(len(rows)/batch));started=time.time()
    # Calibrate a cheap-classifier gate on all calibration pairs, retaining the
    # same low-probability fallback rule for validation and production.
    gate=.02;calindices=np.flatnonzero(calrows&(baseprob>=gate))
    best=None
    for epoch in range(2):
        model.train();loss_sum=0
        for step,start in enumerate(range(0,len(rows),batch)):
            idx=rows[start:start+batch];a,b=pair_texts(pairs.iloc[idx],s1,feed)
            inputs=tokenizer(a,b,padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda');labels=torch.as_tensor(y[idx],device='cuda',dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(**inputs).logits[:,0];loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),labels)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(optimizer);scaler.update();loss_sum+=loss.item()
            if step%100==0:event('ce_training',epoch=epoch+1,step=step,steps=steps,loss=loss_sum/(step+1),elapsed=time.time()-started,peak_vram=torch.cuda.max_memory_allocated()/2**30)
        cp=np.zeros(len(pairs),np.float32);cp[calindices]=ce_predict(model,tokenizer,pairs.iloc[calindices],s1,feed)
        choices=[]
        for weight in [0.,.25,.5,.75,1.]:
            prob=weight*baseprob+(1-weight)*cp;config,score=tune(prob,pairs,y,truthcounts,calmask,countries);choices.append((score,weight,config))
        score,weight,config=max(choices,key=lambda a:a[0]);event('ce_epoch_calibration',epoch=epoch+1,score=score,xgb_weight=weight,thresholds=config)
        if best is None or score>best['calibration_score']:
            best={'calibration_score':score,'xgb_weight':weight,'thresholds':config,'gate':gate,'epoch':epoch+1,'model':MODEL,'frozen_layers':8}
            model.save_pretrained(directory/'best');tokenizer.save_pretrained(directory/'best');np.save(directory/'calibration_probabilities.npy',cp)
            atomic_json(result_path,best)
        else:break
        if score>=.98:break
    del model;gc.collect();torch.cuda.empty_cache();return best


def development():
    status('validation_setup',next_step='reuse embeddings; semantic K20/30/40 + structural candidates')
    s1=pd.read_parquet(ROOT/'cache/normalized/e21_lgbm_c449c0c6b281_source1.parquet')
    feed=pd.read_parquet(ROOT/'cache/normalized/e21_lgbm_c449c0c6b281_feed.parquet')
    _,truth=_load_truth(DEFAULT_CONFIG,set(s1.entity_id))
    q=np.load(ROOT/'cache/embeddings/e26_multilingual_e5_source1_f16.npy',mmap_mode='r');f=np.load(ROOT/'cache/embeddings/e26_multilingual_e5_feed_f16.npy',mmap_mode='r')
    assert q.shape==(len(s1),768) and f.shape==(len(feed),768)
    # Verify cached row mapping against stored E5 retrieval scores.
    old=pd.read_parquet(ROOT/'cache/candidates/e26_multilingual_e5_top20.parquet').iloc[:100]
    qi=pd.Index(s1.entity_id).get_indexer(old.source1_entity_id);fi=pd.Index(feed.entity_id).get_indexer(old.candidate_entity_id)
    check=np.einsum('ij,ij->i',np.asarray(q[qi],np.float32),np.asarray(f[fi],np.float32))
    assert np.max(abs(check-old.e5_similarity.to_numpy()))<.0015,'Embedding row alignment failed'
    fractions=np.array([stable_fraction(value,SEED) for value in s1.entity_id])
    train=fractions<.6;cal=(fractions>=.6)&(fractions<.8);hold=fractions>=.8
    split=pd.DataFrame({'entity_id':s1.entity_id,'country':s1.country_norm,'split':np.where(train,'train',np.where(cal,'calibration','holdout'))});save_frame(split,WORK/'entity_split.parquet')
    event('clean_entity_split',counts=split.groupby(['split','country']).size().to_dict().__str__(),train=int(train.sum()),calibration=int(cal.sum()),holdout=int(hold.sum()))
    ppath=WORK/'development_candidates.parquet'
    if ppath.exists():pairs=pd.read_parquet(ppath)
    else:pairs=candidates(s1,feed,q,f,k=40);save_frame(pairs,ppath)
    for k in [20,30,40]:
        subset=pairs.loc[(pairs.e5_rank<=k)|(pairs.e5_rank==0)|(pairs[['name_full','name_core','address_full','rare_address']].sum(axis=1)>0)]
        event('retrieval_ablation_calibration',k=k,**candidate_metrics(s1.loc[cal,'entity_id'].tolist(),truth,subset.loc[cal[subset.qi.to_numpy()]]))
    featpath=WORK/'development_features.parquet'
    features=pd.read_parquet(featpath) if featpath.exists() else build_features(pairs,s1,feed)
    if not featpath.exists():save_frame(features,featpath)
    y=labeled(pairs,truth);counts=np.array([len(truth.get(s,set())) for s in s1.entity_id]);qc=s1.country_norm.to_numpy()
    tr=train[pairs.qi.to_numpy()];ca=cal[pairs.qi.to_numpy()]
    event('class_balance',train_positive=int(y[tr].sum()),train_negative=int(tr.sum()-y[tr].sum()),cal_positive=int(y[ca].sum()),cal_negative=int(ca.sum()-y[ca].sum()))
    status('gpu_xgboost_training',next_step='calibration threshold tuning and cross-encoder pilot if below target')
    model=train_xgb(features,y,tr,ca,WORK/'encoder_xgb.json');prob=xgb_predict(model,features)
    thresholds,score=tune(prob,pairs,y,counts,cal,qc)
    cfg={'architecture':'e5_structural_xgb','top_k':40,'thresholds':thresholds,'calibration_score':score,'features':features.columns.tolist(),'xgb_weight':1.,'model':MODEL,'split_seed':SEED}
    semantic_threshold,semantic_score=tune(pairs.e5_similarity.to_numpy(),pairs,y,counts,cal,qc)
    event('semantic_only_calibration',score=semantic_score,thresholds=semantic_threshold)
    event('xgb_calibration',score=score,metrics=report(prob,pairs,s1,truth,cal,thresholds),average_precision=average_precision_score(y[ca],prob[ca]),roc_auc=roc_auc_score(y[ca],prob[ca]),logloss=log_loss(y[ca],prob[ca]))
    cfg['model_file']='encoder_xgb.json';cfg['address_view']=False
    atomic_json(WORK/'selection.json',cfg);save_frame(pd.DataFrame({'qi':pairs.qi,'fi':pairs.fi,'label':y,'xgb_probability':prob}),WORK/'development_probabilities.parquet')
    if score<.98:
        status('address_encoder_retrieval_experiment',best_calibration_score=score,next_step='address-only E5 top20 complementary retrieval and GPU classifier')
        apath=WORK/'development_candidates_address.parquet'
        if apath.exists():augmented=pd.read_parquet(apath)
        else:
            augmented=add_address_view(pairs,s1,feed,q,f)
            save_frame(augmented,apath)
        event('address_encoder_retrieval_calibration',**candidate_metrics(s1.loc[cal,'entity_id'].tolist(),truth,augmented.loc[cal[augmented.qi.to_numpy()]]))
        afpath=WORK/'development_features_address.parquet'
        af=pd.read_parquet(afpath) if afpath.exists() else build_features(augmented,s1,feed)
        if not afpath.exists():save_frame(af,afpath)
        ay=labeled(augmented,truth);atr=train[augmented.qi.to_numpy()];aca=cal[augmented.qi.to_numpy()]
        am=train_xgb(af,ay,atr,aca,WORK/'encoder_xgb_address.json');ap=xgb_predict(am,af)
        at,asc=tune(ap,augmented,ay,counts,cal,qc)
        event('address_encoder_xgb_calibration',score=asc,metrics=report(ap,augmented,s1,truth,cal,at))
        if asc>score:
            pairs,features,y,prob=augmented,af,ay,ap
            score=asc;cfg.update({'calibration_score':score,'thresholds':at,'features':af.columns.tolist(),'address_view':True,'model_file':'encoder_xgb_address.json'})
            atomic_json(WORK/'selection.json',cfg)

    status('classifier_validated',best_calibration_score=score,best_model='encoder-aware GPU XGBoost',next_step='cross-encoder if calibration below .98')
    if score<.98:
        ce=train_ce(pairs,s1,feed,y,train,cal,prob,counts,qc)
        if ce['calibration_score']>score:
            cfg.update(ce);cfg['architecture']='e5_structural_xgb_cross_encoder';cfg['features']=features.columns.tolist();atomic_json(WORK/'selection.json',cfg)
    # Selection ends here. Only this frozen configuration sees holdout labels.
    hp=prob.copy()
    if cfg['xgb_weight']<1:
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        ce_model=AutoModelForSequenceClassification.from_pretrained(WORK/'cross_encoder/best').to('cuda');tok=AutoTokenizer.from_pretrained(WORK/'cross_encoder/best')
        use=hold[pairs.qi.to_numpy()]&(prob>=cfg['gate']);cp=np.zeros(len(pairs),np.float32);cp[use]=ce_predict(ce_model,tok,pairs.loc[use],s1,feed)
        hp=cfg['xgb_weight']*prob+(1-cfg['xgb_weight'])*cp
        del ce_model;gc.collect();torch.cuda.empty_cache()
    result=report(hp,pairs,s1,truth,hold,cfg['thresholds']);atomic_json(WORK/'clean_holdout_metrics.json',result)
    event('FROZEN_CLEAN_HOLDOUT',**result)
    status('model_frozen',latest_clean_holdout_score=result['macro_f0_5'],best_calibration_score=cfg['calibration_score'],best_model=cfg['architecture'],reached_0_98_clean_holdout=result['macro_f0_5']>=.98,next_step='full-test embeddings, retrieval, scoring and validation')
    return cfg


def full_test(cfg):
    from sentence_transformers import SentenceTransformer
    from .production_submission import _partition_normalized_test
    manifest=_partition_normalized_test(DEFAULT_CONFIG,ROOT/'cache/production/e23_ensemble_production_v1')
    xmodel=xgb.Booster(model_file=WORK/cfg.get('model_file','encoder_xgb.json'));ce_model=None;tokenizer=None
    if cfg['xgb_weight']<1:
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        ce_model=AutoModelForSequenceClassification.from_pretrained(WORK/'cross_encoder/best').to('cuda');tokenizer=AutoTokenizer.from_pretrained(WORK/'cross_encoder/best')
    started=time.time();stats={}
    identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    frozen_path=WORK/'production_configuration.json'
    if frozen_path.exists():
        assert json.load(frozen_path.open())['sha256']==identity,'Production config changed: use a new checkpoint directory'
    else:atomic_json(frozen_path,{'sha256':identity,'configuration':cfg})
    for country in sorted(manifest['sources']['source1']['paths']):
        country_dir=WORK/'full'/country;country_dir.mkdir(parents=True,exist_ok=True)
        import pyarrow.parquet as pq
        total=pq.ParquetFile(manifest['sources']['source1']['paths'][country]).metadata.num_rows
        checkpoints=[country_dir/f'shard_{start:08d}' for start in range(0,total,5000)]
        if all(p.with_suffix('.json').exists()
               and json.load(p.with_suffix('.json').open()).get('complete')
               and p.with_suffix('.outputs.parquet').exists()
               and pq.ParquetFile(p.with_suffix('.outputs.parquet')).metadata.num_rows==min(5000,total-i*5000)
               for i,p in enumerate(checkpoints)):
            event('full_country_reused',country=country,source1_rows=total,shards=len(checkpoints))
            stats[country]={'source1_rows':total,'reused':True}
            continue
        s1=pd.read_parquet(manifest['sources']['source1']['paths'][country]);feed=pd.concat([pd.read_parquet(manifest['sources'][source]['paths'][country]) for source in ['source2','source3']],ignore_index=True)
        country_start=time.time();status('full_test_country',country=country,source1_rows=len(s1),feed_rows=len(feed))
        encoder=SentenceTransformer(snapshot(),device='cuda');encoder.max_seq_length=128
        batch=encode_benchmark(encoder,feed) if country=='france' else 512
        q=encode(s1,'query',country_dir/'source1.npy',encoder,batch);f=encode(feed,'passage',country_dir/'feed.npy',encoder,batch)
        qa=fa=None
        if cfg.get('address_view'):
            sq=s1.copy();ff=feed.copy();sq['name_full']='';ff['name_full']=''
            qa=encode(sq,'query',country_dir/'address_source1.npy',encoder,batch);fa=encode(ff,'passage',country_dir/'address_feed.npy',encoder,batch)
            del sq,ff
        del encoder;gc.collect();torch.cuda.empty_cache()
        index=StructuralIndex(feed)
        for start in range(0,len(s1),5000):
            end=min(start+5000,len(s1));prefix=country_dir/f'shard_{start:08d}'
            done=prefix.with_suffix('.json')
            if done.exists() and json.load(done.open()).get('complete'):
                event('full_shard_reused',country=country,start=start,end=end);continue
            bt=time.time();subset=s1.iloc[start:end].reset_index(drop=True);pp=prefix.with_suffix('.candidates.parquet')
            if pp.exists():pairs=pd.read_parquet(pp)
            else:
                pairs=candidates(subset,feed,q[start:end],f,k=cfg['top_k'],index=index)
                if cfg.get('address_view'):pairs=add_address_view(pairs,subset,feed,q[start:end],f,qa=qa[start:end],fa=fa)
                save_frame(pairs,pp)
            fp=prefix.with_suffix('.features.parquet')
            retrieval_seconds=time.time()-bt;feature_start=time.time()
            features=pd.read_parquet(fp) if fp.exists() else build_features(pairs,subset,feed)
            if not fp.exists():save_frame(features,fp)
            assert features.columns.tolist()==cfg['features']
            feature_seconds=time.time()-feature_start;inference_start=time.time()
            prob=xgb_predict(xmodel,features)
            if ce_model is not None:
                use=prob>=cfg['gate'];cp=np.zeros(len(pairs),np.float32);cp[use]=ce_predict(ce_model,tokenizer,pairs.loc[use],subset,feed)
                prob=cfg['xgb_weight']*prob+(1-cfg['xgb_weight'])*cp
            inference_seconds=time.time()-inference_start
            selected=decode(prob,pairs,subset,cfg['thresholds'])
            pred=pairs.loc[selected].groupby('source1_entity_id',sort=False).candidate_entity_id.agg(','.join)
            cand=pairs.groupby('source1_entity_id',sort=False).candidate_entity_id.agg(','.join)
            rows=pd.DataFrame({'source1_entity_id':subset.entity_id,'matched_entity_ids':subset.entity_id.map(pred).fillna(''),'candidate_entity_ids':subset.entity_id.map(cand).fillna(''),'_row_order':subset['_row_order']})
            save_frame(rows,prefix.with_suffix('.outputs.parquet'));save_frame(pd.DataFrame({'source1_entity_id':pairs.source1_entity_id,'candidate_entity_id':pairs.candidate_entity_id,'probability':prob}),prefix.with_suffix('.probabilities.parquet'))
            record={'complete':True,'source1_rows':len(subset),'candidate_pairs':len(pairs),'predicted_pairs':int(selected.sum()),'empty_predictions':int(rows.matched_entity_ids.eq('').sum()),'seconds':time.time()-bt,'retrieval_seconds':retrieval_seconds,'feature_seconds':feature_seconds,'inference_seconds':inference_seconds,'peak_ram_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30}
            atomic_json(done,record);event('full_shard_complete',country=country,start=start,end=end,**record)
            status('full_test_inference',country=country,completed_source1_in_country=end,country_source1=len(s1),next_step='continue checkpointed shards')
            del pairs,features,rows;gc.collect()
        stats[country]={'seconds':time.time()-country_start,'source1_rows':len(s1),'feed_rows':len(feed)}
        del s1,feed,index,q,f,qa,fa;gc.collect();torch.cuda.empty_cache()
    atomic_json(WORK/'production_timing.json',{'countries':stats,'seconds':time.time()-started})


def assemble_validate_package():
    status('assemble_and_validate',next_step='official standard + ID validation, then official package')
    parts=[pd.read_parquet(p) for p in sorted((WORK/'full').glob('*/shard_*.outputs.parquet'))]
    rows=pd.concat(parts,ignore_index=True).sort_values('_row_order',kind='stable');assert len(rows)==1732544 and rows.source1_entity_id.is_unique
    assert rows._row_order.tolist()==list(range(1732544))
    out=ROOT/'output';out.mkdir(exist_ok=True)
    for name,cols in [('matching_results.tsv',['source1_entity_id','matched_entity_ids']),('candidate_pairs.tsv',['source1_entity_id','candidate_entity_ids'])]:
        temp=out/(name+'.tmp');rows[cols].to_csv(temp,sep='\t',index=False,chunksize=100000);temp.replace(out/name)
    command=[sys.executable,str(ROOT/'student_resource/utils/validate_submission.py'),'--matching',str(out/'matching_results.tsv'),'--candidate',str(out/'candidate_pairs.tsv'),'--test-dir',str(ROOT/'student_resource/dataset/test')]
    for extra,label in [([], 'standard'),(['--check-ids'],'ids')]:
        proc=subprocess.run(command+extra,capture_output=True,text=True);(LOG/f'encoder_validator_{label}.log').write_text(proc.stdout+'\n'+proc.stderr)
        event('official_validator',kind_of_check=label,returncode=proc.returncode,output=proc.stdout)
        if proc.returncode!=0:raise RuntimeError(f'Validator {label} failed; inspect logs/encoder_validator_{label}.log')
    result=json.load((WORK/'clean_holdout_metrics.json').open());cfg=json.load((WORK/'selection.json').open())
    pairs=int(rows.candidate_entity_ids.str.count(',').add(rows.candidate_entity_ids.ne('').astype(int)).sum());links=int(rows.matched_entity_ids.str.count(',').add(rows.matched_entity_ids.ne('').astype(int)).sum())
    methodology=f'''# Business Entity Resolution — Encoder Submission\n\nTeam name and members: to be completed by submitter.\n\n## Method\nMultilingual E5 base (MIT) encodes normalized business names, addresses and open-set countries. Same-country blocked GPU cosine retrieves top {cfg['top_k']} with bounded exact-name/address and rare-address-token recovery. A GPU XGBoost classifier uses encoder, sparse lexical and numeric features. Cross-encoder configuration: {cfg.get('xgb_weight',1)} XGBoost blend weight.\n\n## Validation\nSource1 entities are deterministically assigned to 60% training, 20% calibration and 20% untouched holdout (seed {SEED}). Thresholds and model choice use calibration only. Frozen holdout macro F0.5: {result['macro_f0_5']:.6f}. This small sampled feed is not a leaderboard estimate. France has no labels and uses the pooled global threshold.\n\n## Results\nFull test Source1 rows: {len(rows)}. Candidate pairs: {pairs}. Predicted links: {links}. Both official validator modes passed. Validator measures validity, not hidden-test F0.5.\n\n## Reproduction\nRun `PYTHONPATH=code/business_entity_resolution/src python -m business_entity_resolution.encoder_pipeline --stage all` from the repository root with official datasets in student_resource/dataset. Cached validation-universe preparation uses the existing stress_test pipeline; see source README. Encoder retrieval and classifier configuration are included below code/business_entity_resolution/artifacts. No external business data was used.\n'''
    (WORK/'Documentation_template.md').write_text(methodology)
    archive=ROOT/'encoder_submission.zip';temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in ['matching_results.tsv','candidate_pairs.tsv']:z.write(out/name,'output/'+name)
        base=ROOT/'code/business_entity_resolution'
        for p in (base/'src').rglob('*.py'):z.write(p,str(p.relative_to(ROOT)))
        for name in ['README.md','requirements.txt']:z.write(base/name,str((base/name).relative_to(ROOT)))
        z.write(WORK/'Documentation_template.md','Documentation_template.md')
        for name in ['selection.json',cfg.get('model_file','encoder_xgb.json'),str(Path(cfg.get('model_file','encoder_xgb.json')).with_suffix('.metadata.json')),'clean_holdout_metrics.json']:
            z.write(WORK/name,'code/business_entity_resolution/artifacts/'+name)
        if cfg['xgb_weight']<1:
            for p in (WORK/'cross_encoder/best').iterdir():
                if p.is_file():z.write(p,'code/business_entity_resolution/artifacts/cross_encoder/'+p.name)
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;(LOG/'encoder_archive_contents.txt').write_text('\n'.join(z.namelist()))
    summary={'clean_macro_f0_5':result['macro_f0_5'],'source1_rows':len(rows),'candidate_pairs':pairs,'predicted_links':links,'empty_predictions':int(rows.matched_entity_ids.eq('').sum()),'total_seconds':time.time()-STATE.get('started_at_epoch',START),'peak_ram_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30,'validator':'PASS','id_check':'PASS','archive':str(archive),'archive_bytes':archive.stat().st_size,'output_sizes':{p.name:p.stat().st_size for p in [out/'matching_results.tsv',out/'candidate_pairs.tsv']}}
    atomic_json(LOG/'ENCODER_FINAL_REPORT.json',summary);status('COMPLETE',**summary,next_step='upload matching_results.tsv and submit archive');event('ENCODER_SUBMISSION_COMPLETE',**summary)
    from .encoder_delivery import deliver
    deliver(ROOT)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','development','full','package'],default='all');args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True);LOG.mkdir(exist_ok=True)
    # Held until process exit, including after the interactive session disconnects.
    lock=(WORK/'pipeline.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('An encoder pipeline already holds the production lock')
    if (WORK/'state.json').exists():STATE.update(json.load((WORK/'state.json').open()))
    try:
        hardware()
        bundled=Path(__file__).resolve().parents[2]/'artifacts'
        if not (WORK/'selection.json').exists() and bundled.exists():
            for item in bundled.iterdir():
                destination=WORK/item.name
                if item.is_dir():
                    shutil.copytree(item,WORK/'cross_encoder/best',dirs_exist_ok=True)
                else:shutil.copy2(item,destination)
        if args.stage in ['all','development']:
            if not (WORK/'clean_holdout_metrics.json').exists():development()
            if args.stage=='development':return
        cfg=json.load((WORK/'selection.json').open())
        if args.stage in ['all','full']:full_test(cfg)
        if args.stage in ['all','full','package']:assemble_validate_package()
    except Exception as error:
        STATE.setdefault('failures',[]).append(str(error));status('FAILED_REQUIRES_FIX',error=str(error),next_step='read traceback, fix and rerun; completed checkpoints preserved');traceback.print_exc();raise

if __name__=='__main__':main()
