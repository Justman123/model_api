from fastapi import FastAPI
import numpy as np
np.bool = np.bool_
import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import gluonnlp as nlp

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


# http로 파일 다운
import requests

# Google Drive 다운로드 링크 리스트
links = [
    "https://drive.google.com/uc?id=115TKxODj7F8oUyxwSBIJCF2md90cc2e8&export=download",
    "https://drive.google.com/uc?id=1dRPT9QBzBMuKUAEOx6lVqRI-vbylwDnL&export=download",
    "https://drive.google.com/uc?id=19sWrbp_oUFETVATh2T3qziSOZKyT7D64&export=download",
    "https://drive.google.com/uc?id=1pBYSD4VO7hsL0sMzakH7_FCHlaK-brL7&export=download"
]

# 저장될 파일 이름 리스트 (i = 2, 3, 4, 5)
file_names = [f"chunk_{i}.bin" for i in range(2, 6)]

# 반복문을 통해 파일 다운로드
for link, file_name in zip(links, file_names):
    print(f"Downloading {file_name} from {link}...")
    response = requests.get(link, stream=True)

    if response.status_code == 200:
        with open(file_name, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"{file_name} downloaded successfully!")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")

print("All downloads completed!")

parts1 = ['chunk_0.bin', 'chunk_1.bin', 'chunk_2.bin', 'chunk_3.bin', 'chunk_4.bin', 'chunk_5.bin']
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
