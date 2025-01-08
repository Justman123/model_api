from fastapi import FastAPI
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

# 압축 파일 병합
parts1 = ['model_pt_folder/chunk_0.bin', 'model_pt_folder/chunk_1.bin', 'model_pt_folder/chunk_2.bin', 'model_pt_folder/chunk_3.bin']
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
