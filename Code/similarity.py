import numpy as np
import private_data
from supabase import create_client, Client
from openai import OpenAI
import tokenizer

oclient = OpenAI(api_key=private_data.aikey)                                                  # openai 클라이언트 생성
Client = create_client(private_data.dburl, private_data.dbkey)                                # 데이터베이스 클라이언트 생성

def get_embedding_list(texts, model="text-embedding-3-large"):                                # openai 임베딩 모델을 사용하여 토큰화된 문자열 임베딩
   embedded = []
   for i in range(len(texts)):
      embedded.append(oclient.embeddings.create(input = texts, model=model).data[0].embedding)
   return embedded

def db_process(dbex):                                                                         # 데이터베이스에 저장된 카테고리 임베딩 데이터 처리
    a = dbex.replace('[', '')
    a = a.replace(']', '')
    a = a.split(',')
    for i in range(len(a)):
      a[i] = float(a[i])
    return a

def cosine_similarity(A, B):                                                                  # 두 임베딩 데이터 간 코사인 유사도 계산
  return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

category_data = Client.table("category").select("*").execute().data                           # 데이터베이스로부터 카테고리 임베딩 데이터 불러오기

while 1:
  emb = get_embedding_list(tokenizer.main())                                                  # tokenizer.py 내 main 함수 호출

  d = {}
  for i in range(len(emb)):                                                                   # 토큰화된 자연어와 카테고리의 유사도 합산
      for j in range(len(category_data)):
        if i == 0:
            d[category_data[j]['title']] = cosine_similarity(emb[i], db_process(category_data[j]['embedding'])) # 딕셔너리 내 키 선언
        d[category_data[j]['title']] += cosine_similarity(emb[i], db_process(category_data[j]['embedding']))    # 코사인 유사도 합산

  for j in range(len(category_data)):
    d[category_data[j]['title']] /= len(emb)                                                  # 합산한 유사도 평균 계산

  d = sorted(d.items(), key=lambda x: x[1], reverse=True)                                     # 유사도 순으로 정렬

  print(d)