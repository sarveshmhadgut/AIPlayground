from dotenv import load_dotenv
from termcolor import colored
from rich.console import Console
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
console = Console()
loader = PyPDFLoader(
    file_path="./DocumentLoaders/docs/Natural Language Processing Tutorial.pdf"
)

docs = loader.load()

text = ""
for i in range(15, 17):
    curr = docs[i]
    text = text + curr.page_content
    print(
        colored("\nPage Content:\n", "magenta"),
        colored(curr.page_content, "yellow"),
        colored("\nMetadata:", "magenta"),
    )
    console.print_json(data=curr.metadata)


# chain
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.7)
parser = StrOutputParser()
prompt = PromptTemplate(
    template="Summarize the text:{text} in about 50 words in clena simple text",
    input_variables=["news"],
)

chain = prompt | model | parser

with console.status(
    "[bold magenta]Generating summary...[/bold magenta]",
    spinner="dots",
):
    res = chain.invoke({"text": text})
print(colored("\nSummary:\n", "magenta"), colored(f"\t{res}", "yellow"))
