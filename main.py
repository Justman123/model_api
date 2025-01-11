from fastapi import FastAPI
import numpy as np
np.bool = np.bool_
import torch
import gluonnlp as nlp
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, tqdm_notebook
from tqdm.notebook import tqdm
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import shutil

# 압축 파일 병합 함수
def merge_files(output_path, input_parts):
    with open(output_path, 'wb') as merged_file:
        for part in input_parts:
            with open(part, 'rb') as part_file:
                merged_file.write(part_file.read())

# API Routing
app = FastAPI()
@app.get("/")
def read_root():
    return {"Let's Start!"}

@app.get("/model")
def read_root(query: str):
    return {"query" : query}

total, used, free = shutil.disk_usage('/')
print(total, used, free)

parts1 = ['chunk_0.bin']
merge_files('model_state_dict.pt', parts1)

parts2 = ['bertmodel_folder/chunk_0.bin', 'bertmodel_folder/chunk_1.bin', 'bertmodel_folder/chunk_2.bin', 'bertmodel_folder/chunk_3.bin']
merge_files('kobert_base_v1/model.safetensors', parts2)


# bertmodel & tokenizer 다운로드 
print('bertmodel 다운 중...')
bertmodel = BertModel.from_pretrained("kobert_base_v1", return_dict=False)
print('bertmodel 다운 완료!')

print('tokenizer 다운 중...')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize
print('tokenizer 다운 완료!')

# model_state_dict.pt 다운로드
print("model_state_dict 다운 중..")

print("model_state_dict 다운 완료..")


# Bert DataSet & Classifier

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset] # 문장 변환
        self.labels = [np.int32(i[label_idx]) for i in dataset] # label 변환

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self): # 전체 데이터셋의 길이 반환
        return (len(self.labels))

class BERTClassifier(nn.Module):
  def __init__(self, bert, hidden_size=768, num_classes=5, dr_rate=None, params=None):
    super(BERTClassifier, self).__init__()
    self.bert = bert
    self.dr_rate = dr_rate

    self.classifier = nn.Linear(hidden_size , num_classes)
    if dr_rate:
      self.dropout = nn.Dropout(p=dr_rate)

  def gen_attention_mask(self, token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
      attention_mask[i][:v] = 1
    return attention_mask.float()

  def forward(self, token_ids, valid_length, segment_ids):
    attention_mask = self.gen_attention_mask(token_ids, valid_length)
    _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
    if self.dr_rate:
        out = self.dropout(pooler)
    else:
        out = pooler
    return self.classifier(out)


# load model
print("model 불러오는 중..")
device = torch.device("cpu")
loaded_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
loaded_model.load_state_dict(torch.load("model_state_dict.pt", weights_only=True, map_location=torch.device('cpu')))
print("model 불러옴!")
