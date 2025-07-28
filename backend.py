from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
import torch, faiss, numpy as np

app = FastAPI(title="Legal AI Backend")

# --- Load Models ---
EMB_MODEL = "nlpaueb/legal-bert-base-uncased"
QA_MODEL = "nlpaueb/legal-bert-small-uncased-finetuned-qa"

embed_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)
embed_model = AutoModel.from_pretrained(EMB_MODEL)

qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)

# --- Load Documents ---
with open("sample_docs.txt", "r") as f:
    documents = [line.strip() for line in f if line.strip()]

# --- Embedding + FAISS ---
def embed_text(text):
    inputs = embed_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)[0].numpy()

doc_embeddings = np.vstack([embed_text(doc) for doc in documents])
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# --- QA helper ---
def extract_answer(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    return qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start:end]))

# --- API Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

@app.post("/search")
def search_docs(request: QueryRequest):
    qvec = embed_text(request.query).reshape(1, -1)
    distances, indices = index.search(qvec, request.top_k)
    results = [{"doc": documents[i], "score": float(distances[0][idx])} for idx, i in enumerate(indices[0])]
    return {"query": request.query, "results": results}

@app.post("/qa")
def qa_legal(request: QueryRequest):
    qvec = embed_text(request.query).reshape(1, -1)
    distances, indices = index.search(qvec, request.top_k)
    answers = []
    for i in indices[0]:
        context = documents[i]
        ans = extract_answer(request.query, context)
        answers.append({"context": context, "answer": ans})
    return {"query": request.query, "answers": answers}
