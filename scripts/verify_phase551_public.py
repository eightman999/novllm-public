#!/usr/bin/env python3
"""Recompute published BPB and work-bootstrap intervals without private inputs."""
import argparse,csv,json,math,random
from pathlib import Path

def load(path):
    return list(csv.DictReader(path.open()))

def verify(audit):
    audit=Path(audit);scores=load(audit/'work_scores.csv');totals=load(audit/'evaluation_summary.csv');comparisons=load(audit/'hardened_comparison.csv')
    for row in totals:
        group=[s for s in scores if all(s[k]==row[k] for k in ('dataset','seed','candidate','category'))]
        nll=sum(float(s['nll_sum']) for s in group);denominator=sum(int(s['source_bytes']) for s in group)
        assert denominator==int(row['source_bytes'])
        assert math.isclose(nll/denominator/math.log(2),float(row['bits_per_byte']),rel_tol=1e-12)
    for row in comparisons:
        pair=[]
        for size in ('32k','48k'):
            group=sorted([s for s in scores if s['candidate']=='j-reversible-sp-unigram-'+size and all(s[k]==row[k] for k in ('dataset','seed','category'))],key=lambda s:int(s['bootstrap_work_index']))
            assert len(group)==int(row['works'])
            assert [int(s['bootstrap_work_index']) for s in group]==list(range(len(group)))
            pair.append(group)
        assert [s['source_work_identity'] for s in pair[0]]==[s['source_work_identity'] for s in pair[1]]
        assert [s['source_bytes'] for s in pair[0]]==[s['source_bytes'] for s in pair[1]]
        rng=random.Random(int(row['bootstrap_seed']));samples=[]
        for _ in range(int(row['bootstrap_replicates'])):
            indices=rng.choices(range(len(pair[0])),k=len(pair[0]))
            nll=[sum(float(group[i]['nll_sum']) for i in indices) for group in pair]
            samples.append(100*(nll[1]/nll[0]-1))
        samples.sort()
        for actual,expected in [(samples[49],float(row['ci_lower_percent'])),(samples[1949],float(row['ci_upper_percent']))]:
            assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10),(actual,expected)
    return {'status':'PASS','domain_totals_recomputed':len(totals),'bootstrap_intervals_recomputed':len(comparisons),'private_data_required':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    result=verify(a.audit);print(json.dumps(result))
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
