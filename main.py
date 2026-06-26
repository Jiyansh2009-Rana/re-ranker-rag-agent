import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import requests
from supabase import create_client, Client
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from io import BytesIO
import PyPDF2
from dotenv import load_dotenv
import uvicorn

load_dotenv()


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY =  os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


app = FastAPI()

def get_jina_embeddings(text: str) -> List[float]:
    
    url = "https://api.jina.ai/v1/embeddings"
    
    headers = {
               "Content-Type":"application/json",
                "Authorization": f"Bearer {JINA_API_KEY}"
            }
                
    data = {
             "model":"jina-embeddings-v3",
             "dimensions": 1024,
             "input": [text]
            }
            
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        raise Exception(f"Jina AI Error: {response.text}")
        
    return response.json()["data"][0]["embedding"]
    
    
    
def reranking(query : str, document : str):
    
    url = "https://api.jina.ai/v1/rerank"
    
    headers = {
               "Content-Type":"application/json",
                "Authorization": f"Bearer {JINA_API_KEY}"
            }

    data = {
             "model":"jina-reranker-v3",
             "query": query,
             "top_n": 3,
             "documents": [document],
             "return_documents": True
            }    
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        raise Exception(f"Jina AI Error: {response.text}")
    
    return response.json()["results"][0]["document"]["text"]
    
    
    
def pdf_reader(file : bytes) -> str:
    
    reader = PyPDF2.PdfReader(BytesIO(file))
    text = "".join([page.extract_text() or "" for page in reader.pages])
    
    return text
    
@app.post("/Upload-Data")
async def upload_pdf(file : UploadFile = File(...)):
    
    try:
        file_bytes = await file.read()
        
        if file.filename.endswith(".pdf"):
            text = pdf_reader(file_bytes)
            
        else:
         
            raise HTTPException(status_code=400, detail="Only text Based pdf file")
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="File contain is not reliable text")
        
        
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        
        data_to_insert = []
        for chunk in chunks:
            
            embedding = get_jina_embeddings(chunk)
            data_to_insert.append({
            "content": chunk,
            "embedding": embedding,
            "metadata": {"filename": file.filename}
    })

# Single network call instead of many
        if data_to_insert:
            
            
            supabase.table("documents").insert(data_to_insert).execute()

        return {
        "message": f"Successfully processed {len(chunks)} chunks from {file.filename}"
    }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  
        
class query(BaseModel):
    system_prompt: str
    user_query: str
    language:str         
           
           
@app.get("/language-code")
def lang_code():
    return {
            "en" : "English",
            "hi" : "Hindi",
            "fr" : "French",
            "de": "German",
            "it" : "Italian"
     }
     
@app.post("/query")
def query_rag(basemodel : query):
    system_prompt = basemodel.system_prompt
    user_query = basemodel.user_query
    language = basemodel.language  
    
    query_embedding = get_jina_embeddings(user_query)
    
    metadata_filter = {}
    
    rpc_params = {
        "query_text": user_query,
        "query_embedding": query_embedding,
        "match_count": 10,
        "metadata_filter": metadata_filter,
        "vector_weight": 0.7,
        "keyword_weight" :0.3
    }
    results = supabase.rpc("hybrid_search", rpc_params).execute()
    
    context = "\n".join([res["content"] for res in results.data])
    
    re_result = reranking (user_query , context)
    
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", 
            "content": f"{system_prompt}\n\nAnswer in {language}\n\nContext:\n{re_result}"},
            {"role": "user",
             "content": user_query
            }],
        temperature=0.5
    )
    
    answer = completion.choices[0].message.content
    
    return {
        "answer" : answer 
    }
    
                                                                         
                                                                         
      
if __name__ == "__main__" :
                                       uvicorn.run(app,host="localhost",port= 8000)
                                       
                                       
                    
                    
            