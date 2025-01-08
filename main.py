from fastapi import FastAPI
import numpy as np
np.bool = np.bool_
import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import gluonnlp as nlp
import gdown

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

# google drive에서 model.pt 다운로드

# URL 리스트
urls = [
    "https://drive.google.com/uc?id=1vdQ6-eoetgIxNDh8NTAVm9S_Zu3RwBds",
    "https://drive.google.com/uc?id=1Ys10nBs-gvAEetmyjTXLlKp8YZpzCH2H",
    "https://drive.google.com/uc?id=1sGINbXMp3bKmu_5OeGkz8yLpF8m8R4o2",
    "https://drive.google.com/uc?id=1RJo8w01-dZyRDAxKDBCCEzWo64kZupOl",

]

# 외부에서 선언된 i (다운로드 시작 인덱스)
i = 1  # 시작 인덱스를 설정

# model_pt_chunk 파일 다운로드
for url in urls:
    output_file = f"model_pt_folder/chunk_{i}"  # 기본 파일 이름 설정
    print(f"Downloading {url} to {output_file}...")
    try:
        gdown.download(url, output_file, quiet=False)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    i += 1  # 다음 인덱스로 증가

# 압축 파일 병합
parts1 = ['model_pt_folder/chunk_0.bin', 'model_pt_folder/chunk_1.bin', 'model_pt_folder/chunk_2.bin', 'model_pt_folder/chunk_3.bin']
merge_files('model_state_dict.pt', parts1)

parts2 = ['bertmodel_folder/chunk_0.bin', 'bertmodel_folder/chunk_1.bin', 'bertmodel_folder/chunk_2.bin', 'bertmodel_folder/chunk_3.bin']
merge_files('kobert_base_v1/model.safetensors', parts2)

# bertmodel & tokenizer 다운로드 
print('bertmodel 다운 중...')
bertmodel = BertModel.from_pretrained("kobert_base_v1", return_dict=False)
print('bertmodel 다운 완료!')

# print('tokenizer 다운 중...')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize
print('tokenizer 다운 완료!')
