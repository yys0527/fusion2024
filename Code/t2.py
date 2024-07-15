import numpy as np
import private_data
from supabase import create_client, Client
from openai import OpenAI
import re

oclient = OpenAI(api_key=private_data.aikey)

def get_embedding_list(texts, model="text-embedding-3-large"):
   return oclient.embeddings.create(input = texts, model=model).data[0].embedding

def db_process(dbex):
    a = dbex.replace('[', '')
    a = a.replace(']', '')
    a = a.split(',')
    for i in range(len(a)):
      a[i] = float(a[i])
    return a

def cosine_similarity(A, B):
  return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

emb = get_embedding_list("레시피")

Client = create_client(private_data.dburl, private_data.dbkey)
category_data = Client.table("category").select("*").execute().data

d = {}

for i in range(len(category_data)):
    d[category_data[i]['title']] = cosine_similarity(emb, db_process(category_data[i]['embedding']))

d = sorted(d.items(), key=lambda x: x[1], reverse=True)
print(d)