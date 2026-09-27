import numpy as np
import pandas as pd
from business_entity_resolution.encoder_pipeline import metric_arrays,lexical_pair_stats,decode

def test_encoder_metric_empty_and_multimatch():
    counts=np.array([0,2,1]);q=np.array([1,1,2]);labels=np.array([1,1,0]);mask=np.ones(3,bool)
    assert metric_arrays(q,counts,labels,np.array([1,1,0],bool),mask)==2/3
    assert metric_arrays(q,counts,labels,np.zeros(3,bool),mask)==1/3

def test_sparse_pair_token_overlap():
    shared,jacc,_,_=lexical_pair_stats(pd.Series(['blue shop','']),pd.Series(['blue store','']))
    np.testing.assert_allclose(shared,[1,0]);np.testing.assert_allclose(jacc,[1/3,1])

def test_open_set_decoding_and_multiple_matches():
    pairs=pd.DataFrame({'qi':[0,0,1,2]});s1=pd.DataFrame({'country_norm':['india','us','france']})
    actual=decode(np.array([.9,.8,.5,.8]),pairs,s1,{'global':.75,'countries':{'india':.7,'us':.6}})
    assert actual.tolist()==[True,True,False,True]


def test_unique_string_factorization_preserves_sparse_values():
    from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
    left=pd.Series(['alpha alpha ltd','12 rue école','alpha alpha ltd',''])
    right=pd.Series(['alpha ltd','12 rue ecole','',''])
    cv=CountVectorizer(binary=True,lowercase=False,token_pattern=r'(?u)\b\w+\b',dtype=np.float32)
    cv.fit(pd.concat([left,right]))
    a,b=cv.transform(left),cv.transform(right)
    shared=np.asarray(a.multiply(b).sum(axis=1)).ravel()
    actual=lexical_pair_stats(left,right)
    np.testing.assert_array_equal(actual[0],shared)
    hv=HashingVectorizer(analyzer='char',ngram_range=(2,3),n_features=2**18,alternate_sign=False,norm='l2',lowercase=False,dtype=np.float32)
    expected=np.asarray(hv.transform(left).multiply(hv.transform(right)).sum(axis=1)).ravel()
    np.testing.assert_array_equal(lexical_pair_stats(left,right,char=True),expected)


def test_entity_split_is_disjoint_and_deterministic():
    from business_entity_resolution.split import stable_fraction
    from business_entity_resolution.encoder_pipeline import SEED
    ids=['S1-'+str(i) for i in range(500)]
    a=np.array([stable_fraction(v,SEED) for v in ids])
    b=np.array([stable_fraction(v,SEED) for v in ids])
    np.testing.assert_array_equal(a,b)
    train=a<.6;cal=(a>=.6)&(a<.8);hold=a>=.8
    assert not np.any(train&cal) and not np.any(train&hold) and not np.any(cal&hold)
    assert np.all(train|cal|hold)
    pair_entities=np.repeat(np.arange(500),7)
    assert not set(pair_entities[train[pair_entities]])&set(pair_entities[hold[pair_entities]])
