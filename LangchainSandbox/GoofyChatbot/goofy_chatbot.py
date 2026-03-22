import shutil
from termcolor import colored
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
width = shutil.get_terminal_size().columns
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
)

model = ChatHuggingFace(llm=llm)

chat_history = [SystemMessage("Enact Goofy from Mickey Mouse")]

with open("chat_history.txt", "r") as reader:
    chat_history.extend(reader.readlines())

while True:
    print(colored("You: ", "blue"), end="")
    user_query = input()
    if user_query == "exit":
        break

    human_query = HumanMessage(content=user_query)
    chat_history.append(human_query)

    res = model.invoke(chat_history)
    chat_history.append(AIMessage(content=res.content))

    print(f"{colored('buh:', 'green')} {res.content}", end="\n\n")


with open("chat_history.txt", "a") as writer:
    for chat in chat_history:
        writer.write(str(chat) + "\n")
