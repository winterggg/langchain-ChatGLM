import json
import time

from sqlalchemy import JSON
from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
from flask import Flask, request, Response
import uuid
from flask_cors import CORS

app = Flask(__name__)
# 中文乱码
app.config['JSON_AS_ASCII'] = False

CORS(app, supports_credentials=True)



# todo: NLTK 的作用
nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 1

# Show reply with source text from input document
REPLY_WITH_SOURCE = True


local_doc_qa = LocalDocQA()
local_doc_qa.init_cfg(llm_model=LLM_MODEL,
                      embedding_model=EMBEDDING_MODEL,
                      embedding_device=EMBEDDING_DEVICE,
                      llm_history_len=LLM_HISTORY_LEN,
                      top_k=VECTOR_SEARCH_TOP_K)
print("LocalDocQA initialized.")


# generate vs store on certain path
def generate_vs_store(local_doc_qa, corpus_path, store_name):
    store_name = f"{store_name}_{LLM_MODEL}_{EMBEDDING_MODEL}_{int(time.time())}"
    vs_path, _ = local_doc_qa.init_knowledge_vector_store(corpus_path, store_name = store_name)
    return store_name

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('file')
    unique_folder = str(uuid.uuid4())
    corpus_path = os.path.join('./corpus', unique_folder)
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)
    for file in files:
        file.save(os.path.join(corpus_path, file.filename))
    store_name = generate_vs_store(local_doc_qa, corpus_path, "dlw")
    return {'store_name': store_name}

@app.route('/get_answer/<store_name>', methods=['GET'])
def stream_get_answer(store_name):
    query = request.args.get('query')
    vs_path = f"{VS_ROOT_PATH}{store_name}"
    history = []
    res = {}
    for resp, history  in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                vs_path=vs_path,
                                                                chat_history=history,
                                                                streaming=False):
      res["answer"] = resp["result"]
    if REPLY_WITH_SOURCE:
        output = [{
            'name': f'出处 [{inum + 1}] {os.path.split(doc.metadata["source"])[-1]}', 
            'content': doc.page_content
            }   for inum, doc in enumerate(resp["source_documents"])]
        res["source"] = output
        
    return json.dumps(res, ensure_ascii=False)

if __name__ == "__main__":
    # print(generate_vs_store(local_doc_qa, './corpus/bch', "ny"))
    app.run(debug=False, port=5001)