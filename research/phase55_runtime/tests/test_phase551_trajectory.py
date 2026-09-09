import json
import math
import pytest
import torch
import hashlib
import sentencepiece as spm
from canonical_corpus.probe_lm import ProbeLM, LMConfig
from canonical_corpus.tokenizer_adapters import ReversibleSentencePieceAdapter
from canonical_corpus import phase55_same_source_runtime as rt
from scripts.eval_phase551_checkpoints import score_hardened, trajectory_plan, checkpoint_ready, trajectory_comparisons, trajectory_stem



@pytest.fixture
def inputs(tmp_path):
    # Self-contained synthetic text; public tests do not require private fixtures.
    values=[('synthetic-a','validation 日本語 text','kanbun'),
            ('synthetic-b','a  b\n\n▁ \ue000 \x00','kakikudashi')]
    evaluation=[dict(id=i,document_id=i,text=t,category=c,text_sha256=hashlib.sha256(t.encode()).hexdigest()) for i,t,c in values]
    path=tmp_path/'tokenizer'
    spm.SentencePieceTrainer.train(sentence_iterator=iter(ReversibleSentencePieceAdapter.escape_text(r['text']) for r in evaluation),
        model_prefix=str(path),vocab_size=400,minloglevel=2,hard_vocab_limit=False,
        normalization_rule_name='identity',add_dummy_prefix=False,remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,byte_fallback=True,bos_id=1,eos_id=2,unk_id=0,pad_id=-1)
    tokenizer=ReversibleSentencePieceAdapter(path.with_suffix('.model'))
    cfg=LMConfig(vocab_size=tokenizer.processor.get_piece_size(),hidden_size=16,num_layers=1,num_heads=2,num_kv_heads=2,ffn_size=24,context_length=4)
    return {'tokenizer_path':path.with_suffix('.model'),'model_config':cfg,'eval_records':evaluation}


def test_trajectory_scores_match_existing_scorer(inputs):
    rt.set_seed(20260909,threads=1)
    model=ProbeLM(inputs['model_config']).eval()
    tokenizer=ReversibleSentencePieceAdapter(inputs['tokenizer_path'])
    records=inputs['eval_records']
    measured,items=score_hardened(model,tokenizer,records,inputs['model_config'],torch.device('cpu'))
    prepared,_=rt._prepare(tokenizer,records)
    reference=rt.evaluate(model,prepared,inputs['model_config'],torch.device('cpu'),4096)
    assert measured['domains']==reference['domains']
    assert len(items)==len(records)
    assert math.isclose(sum(x['nll_sum'] for x in items),reference['overall']['nll_sum'])
    assert measured['preservation']['unknown_tokens']==0


def test_only_ten_requested_checkpoints_and_complete_sidecar(tmp_path):
    assert len(trajectory_plan())==10
    assert trajectory_plan()[:6]==[(s,f) for s in ('32k','48k') for f in (.2,.5,.75)]
    path=tmp_path/'checkpoint.pt';path.write_bytes(b'fixture')
    assert not checkpoint_ready(path)
    path.with_suffix('.sha256').write_text('a'*12)
    assert not checkpoint_ready(path)
    path.with_suffix('.sha256').write_text('a'*64+'\n')
    assert checkpoint_ready(path)


def test_comparisons_require_same_hardened_identity(tmp_path):
    output=tmp_path/'out';output.mkdir()
    for size,bpb in [('32k',2.),('48k',2.1)]:
        row={'dataset':'hardened','dataset_sha256':'x','seed':1,'checkpoint_sha256':size,
             'domains':{c:{'bits_per_byte':bpb} for c in ('kanbun','kakikudashi')},
             'works':[],'actual_source_chars':6000001}
        (output/(trajectory_stem(size,.2)+'.json')).write_text(json.dumps(row))
    result=trajectory_comparisons(output,tmp_path/'absent')
    assert len(result)==2 and math.isclose(result[0]['relative_percent'],5)
    path=output/(trajectory_stem('48k',.2)+'.json');row=json.loads(path.read_text());row['dataset_sha256']='other';path.write_text(json.dumps(row))
    with pytest.raises(AssertionError):trajectory_comparisons(output,tmp_path/'absent')
