from dotenv import load_dotenv
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    # HuggingFacePipeline,
)

load_dotenv()

# API
llm_api = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=50,
)

model_api = ChatHuggingFace(llm=llm_api)
res_api = model_api.invoke("Who won superbowl 2000?")
print("API:", res_api.content)

#  Local
# llm_local = HuggingFacePipeline.from_model_id(
#     model_id="google/gemma-2b-it",
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 100},
# )
# model_local = ChatHuggingFace(llm=llm_local)
# res_local = model_local.invoke("Who won superbowl 2000?")
# print("Local:", res_local.content)
