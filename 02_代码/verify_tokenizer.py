"""Verify provided IDs against the chosen vocabulary on train/valid only."""
import json
import pickle
from pathlib import Path
import numpy as np
from transformers import BertTokenizer
from p2 import DATA


def verify(data,model_dir):
    tokenizer=BertTokenizer(vocab_file=str(Path(model_dir)/'vocab.txt'),do_lower_case=True)
    report={}
    for split in ('train','valid'):
        part=data[split];exact=0;active=0;matches=0
        for i,text in enumerate(part['raw_text']):
            encoded=np.array(tokenizer(str(text),padding='max_length',truncation=True,max_length=50)['input_ids'])
            provided=part['text_bert'][i,0];mask=part['text_bert'][i,1].astype(bool)
            exact+=int(np.array_equal(encoded,provided));active+=int(mask.sum());matches+=int((encoded[mask]==provided[mask]).sum())
        report[split]={'n':len(part['id']),'exact_sequence_matches':exact,'position_match_rate':matches/active}
        if exact!=len(part['id']):raise ValueError(f'{split}: tokenizer mismatch; do not enable unverified encoder')
    report['vocab_size']=len(tokenizer)
    report['special_ids']={key:getattr(tokenizer,key) for key in ['pad_token_id','unk_token_id','cls_token_id','sep_token_id']}
    if report['special_ids']!={'pad_token_id':0,'unk_token_id':100,'cls_token_id':101,'sep_token_id':102}:
        raise ValueError('Special token mapping differs; change Dataset policy explicitly')
    return report


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    with (DATA/'附件2-数据集特征文件'/'aligned_50.pkl').open('rb') as f:data=pickle.load(f)
    print(json.dumps(verify(data,root/'models/bert_tiny'),ensure_ascii=False,indent=2))
