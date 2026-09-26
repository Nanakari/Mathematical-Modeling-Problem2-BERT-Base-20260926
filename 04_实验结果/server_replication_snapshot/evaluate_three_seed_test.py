"""Post-freeze diagnostic of three predeclared full-model seeds; no selection."""
import csv, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
LANE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import v5_train
from v5_data import AlignedDataset, load_tokenizers
with (LANE/'official_test_only.pkl').open('rb') as f: raw=pickle.load(f)
assert set(raw)=={'test'}
test=AlignedDataset(AlignedDataset._select(raw['test']),labeled=True,source='attachment2/test-postfreeze-diagnostic',special_token_ids=(101,102))
del raw
samples=[test[i] for i in range(len(test))]
y=np.asarray(test.part['classification_labels'],dtype=int)
r=np.asarray(test.part['regression_labels'],dtype=float)
ids=[str(i) for i in test.part['id']]
assert len(ids)==len(set(ids))==727
rows=[{'sample_id':ids[i],'true_class':int(y[i]),'true_intensity':float(r[i])} for i in range(len(y))]
summary={'usage':'postfreeze_diagnostic_only; no model selection','model':'full AV-direct BERT-Base','results':{}}
for seed in (1729,2718,3407):
 if seed==1729:
  cfg=ROOT/'configs/v5/student_bert_base_retest_seed1729.json'
  out=ROOT/'outputs/bert_base_vs_minilm_20260926/student_bert_base_retest_seed1729'
 else:
  cfg=LANE/f'full_seed{seed}.json'; out=LANE/f'full_seed{seed}'
 config=json.loads(cfg.read_text())
 best=json.loads((out/'best_validation.json').read_text())
 model=v5_train._load_best_model(config,out/'best_model.safetensors',device=torch.device('cuda'),with_distill_heads=False)
 norm=v5_train._load_standardizer(out/'scaler.json')
 source_tok,target_tok=load_tokenizers(v5_train.resolve_project_path(config['model']['backbone_dir']),source_vocab=v5_train.source_vocab_from(config))
 metrics,logits,intensity=v5_train._evaluate(model,samples,y,r,norm,source_tok,target_tok,device=torch.device('cuda'),batch_size=32,max_text_length=int(config['data']['max_text_length']),amp_dtype=config['training']['amp_dtype'])
 pred=logits.argmax(axis=1)
 for i,row in enumerate(rows):
  row[f'seed{seed}_pred_class']=int(pred[i]); row[f'seed{seed}_pred_intensity']=float(intensity[i])
 summary['results'][str(seed)]={'validation_best_epoch':best['epoch'],'validation_metrics':best['metrics'],'test_metrics':metrics,'checkpoint':str(out/'best_model.safetensors')}
 print(seed,metrics,flush=True)
 del model; torch.cuda.empty_cache()
with (LANE/'three_seed_test_per_sample.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(LANE/'three_seed_test_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
