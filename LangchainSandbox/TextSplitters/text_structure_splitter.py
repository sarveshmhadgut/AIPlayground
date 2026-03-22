from termcolor import colored
from dotenv import load_dotenv
from rich.console import Console
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
console = Console()

loader = TextLoader(
    file_path="./TextSplitters/docs/andromeda_paradox.txt", encoding="utf-8"
)
docs = loader.load()

page = docs[0].page_content
metadata = docs[0].metadata

print(colored("Page Content:\n", "magenta"), colored(page, "yellow"))
print(colored("Page Metadata:\n", "magenta"), colored(metadata, "yellow"))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = splitter.split_text(page)
print(colored("\nSplits:", "magenta"))
console.print_json(data={"Split[0]": texts[0], "no.of splits": len(texts)})
