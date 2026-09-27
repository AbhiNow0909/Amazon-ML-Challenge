import os,time,json
import pandas as pd
from business_entity_resolution.encoder_pipeline import build_features,WORK,ROOT
if __name__=='__main__':
    s=pd.read_parquet(ROOT/'cache/normalized/e21_lgbm_c449c0c6b281_source1.parquet')
    f=pd.read_parquet(ROOT/'cache/normalized/e21_lgbm_c449c0c6b281_feed.parquet')
    p=pd.read_parquet(WORK/'development_candidates.parquet')
    p=p.loc[p.qi<2000].reset_index(drop=True)
    expected=pd.read_parquet(WORK/'development_features.parquet').iloc[:len(p)].reset_index(drop=True)
    for workers in [1,4,4]:
        os.environ['ENCODER_FEATURE_WORKERS']=str(workers)
        start=time.time();actual=build_features(p,s,f)
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)
        print(json.dumps({'workers':workers,'pairs':len(p),'seconds':time.time()-start,'equality':'EXACT'}),flush=True)
