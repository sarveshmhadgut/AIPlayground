from termcolor import colored
from dotenv import load_dotenv
from rich.console import Console
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()
console = Console()

loader = WebBaseLoader(
    web_path="https://docs.langchain.com/oss/python/integrations/document_loaders/web_base"
)

docs = loader.load()

print(colored("Document Structure", "magenta"))
console.print_json(data=docs[0].model_dump())

content = docs[0].page_content

prompt = PromptTemplate(
    template="Summarize the content of web page: {content} in about 50 words.",
    input_variables=["content"],
)
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.7)
parser = StrOutputParser()

chain = prompt | model | parser
with console.status(
    "[bold magenta]Summarizing Web Content...[/bold magenta]",
    spinner="dots",
):
    res = chain.invoke({"content": content})

print(colored("Summary:\n", "magenta"), colored(res, "yellow"))
