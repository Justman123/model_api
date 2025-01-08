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
urls1 = [
    "https://drive.google.com/uc?id=115TKxODj7F8oUyxwSBIJCF2md90cc2e8",
    "https://drive.google.com/uc?id=1dRPT9QBzBMuKUAEOx6lVqRI-vbylwDnL",
]
# 외부에서 선언된 i (다운로드 시작 인덱스)
i = 2  # 시작 인덱스를 설정

# model_pt_chunk 파일 다운로드
for url in urls1:
    output_file = f"model_pt_folder/chunk_{i}.bin"  # 기본 파일 이름 설정
    print(f"Downloading {url} to {output_file}...")
    try:
        gdown.download(url, output_file, quiet=False)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    i += 1  # 다음 인덱스로 증가

print('1차 다운 완료!')
# 압축 파일 병합
parts2 = ['bertmodel_folder/chunk_0.bin', 'bertmodel_folder/chunk_1.bin', 'bertmodel_folder/chunk_2.bin', 'bertmodel_folder/chunk_3.bin']
merge_files('kobert_base_v1/model.safetensors', parts2)

# model_pt_chunk 파일 다운로드
urls2 = [
    "https://drive.google.com/uc?id=19sWrbp_oUFETVATh2T3qziSOZKyT7D64",
    "https://drive.google.com/uc?id=1pBYSD4VO7hsL0sMzakH7_FCHlaK-brL7",
] 

for url in urls2:
    output_file = f"model_pt_folder/chunk_{i}.bin"  # 기본 파일 이름 설정
    print(f"Downloading {url} to {output_file}...")
    try:
        gdown.download(url, output_file, quiet=False)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    i += 1  # 다음 인덱스로 증가
print('2차 다운 완료!')

parts1 = ['chunk_0.bin', 'chunk_1.bin', 'chunk_2.bin', 'chunk_3.bin', 'chunk_4.bin', 'chunk_5.bin']
merge_files('model_state_dict.pt', parts1)

# bertmodel & tokenizer 다운로드 
print('bertmodel 다운 중...')
bertmodel = BertModel.from_pretrained("kobert_base_v1", return_dict=False)
print('bertmodel 다운 완료!')

# print('tokenizer 다운 중...')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize
print('tokenizer 다운 완료!')
