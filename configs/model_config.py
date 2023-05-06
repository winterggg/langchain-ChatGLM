import torch.cuda
import torch.backends
import os

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
llm_model_dict = {
    "chatyuan": "ClueAI/ChatYuan-large-v2",
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int8": "THUDM/chatglm-6b-int8",
    "chatglm-6b": "THUDM/chatglm-6b",
}

# LLM model name
LLM_MODEL = "chatglm-6b-int4"

# LLM lora path，默认为空，如果有请直接指定文件夹路径
LLM_LORA_PATH = ""
USE_LORA = True if LLM_LORA_PATH else False

# LLM streaming reponse
STREAMING = True

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content")

PROMPT_TEMPLATE = """已知信息：
{context} 

请根据这些信息，以专业且简洁的方式回答下列问题：“{question}”。
在回答问题时，如果某些信息与问题无关，可以选择忽略。如果可以从这些信息中找到答案，请直接回答。如果无法直接从这些信息中找到答案，但可以基于这些信息进行合理的推断，那么请给出推断，并明确指出这是一种推断。
如果这些信息并不足以回答这个问题，那么请回答：“对不起，我无法根据提供的信息回答这个问题。可能需要更多的相关信息。” 请尽量避免在回答中添加任何基于猜测或创作的内容。所有的回答都应该以中文提供。
"""
## 旧版模板
# PROMPT_TEMPLATE = """已知信息：
# {context} 

# 根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

# 匹配后单段上下文长度
CHUNK_SIZE = 250

# LLM input history length
LLM_HISTORY_LEN = 3

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 5

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
