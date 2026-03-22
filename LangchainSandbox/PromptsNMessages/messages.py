from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
messages = [
    SystemMessage("Enact Donald Duck"),
    HumanMessage("Explain RLHF in short"),
]

res = model.invoke(messages)
print(res.content)
