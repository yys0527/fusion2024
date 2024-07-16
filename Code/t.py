import private_data
from supabase import create_client, Client
from openai import OpenAI
Client = create_client(private_data.dburl, private_data.dbkey)


oclient = OpenAI(api_key=private_data.aikey)

def get_embedding_list(texts, model="text-embedding-3-large"):
   return oclient.embeddings.create(input = texts, model=model).data[0].embedding

topic = "사회"
emb = get_embedding_list(topic)
print(Client.table("category").insert({"title": topic, "embedding": emb}).execute())

print(Client.table("category").select("*").execute().data[0])