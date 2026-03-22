from termcolor import colored
from dotenv import load_dotenv
from rich.console import Console
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import PythonCodeTextSplitter

load_dotenv()
console = Console()

loader = TextLoader(file_path="./TextSplitters/docs/python_class.txt", encoding="utf-8")
docs = loader.load()
page, metadata = docs[0].page_content, docs[0].metadata
print(colored("Page Content:\n", "magenta"), colored(page, "yellow"))
print(colored("Page Metadata:\n", "magenta"), colored(metadata, "yellow"))

splitter = PythonCodeTextSplitter(chunk_size=580, chunk_overlap=0)
texts = splitter.split_text(page)
splits = {i: split for i, split in enumerate(texts)}

print(colored("\nSplits:", "magenta"))
console.print_json(data={"Splits": splits, "no.of splits": len(texts)})
