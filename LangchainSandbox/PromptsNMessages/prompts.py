from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=50,
)

model = ChatHuggingFace(llm=llm)

# Template
user_input = input("What's your topic: ")
prompt = PromptTemplate.from_template("Explain {topic} in short")

read_prompt = prompt.invoke({"topic": user_input})
res = model.invoke(read_prompt).content
print("Res: ", res, end="\n\n")

# Chat Template
prompts = ChatPromptTemplate.from_messages(
    [("system", "Enact {role}"), ("human", "Explain {topic} in short")]
)

role = input("Enter role: ")
topic = input("Enter topic: ")

read_prompt = prompts.invoke({"role": role, "topic": topic})
res = model.invoke(read_prompt).content
print("Res: ", res)
