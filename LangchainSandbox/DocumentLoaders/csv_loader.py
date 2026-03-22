from termcolor import colored
from dotenv import load_dotenv
from rich.console import Console
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader

load_dotenv()
console = Console()
loader = CSVLoader(
    file_path="./DocumentLoaders/docs/telco_churn.csv", autodetect_encoding=True
)

# docs
docs = loader.load()
print(colored("Document Structure:", "magenta"))
console.print_json(data=docs[0].model_dump())

clients = [docs[i].page_content.split("\n") for i in range(5)]
print(colored("Clients Structure:", "magenta"))
console.print(clients[0])


model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.7)
prompt = PromptTemplate(
    template="Perform primitive EDA on {clients} in about 50 words.",
    input_variables=["clients"],
)
parser = StrOutputParser()

chain = prompt | model | parser

with console.status(
    "[bold magenta]Performing EDA...[/bold magenta]",
    spinner="dots",
):
    res = chain.invoke({"clients": clients})

print(colored("EDA:\n", "magenta"), colored(res, "yellow"))
