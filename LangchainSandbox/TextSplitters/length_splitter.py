from dotenv import load_dotenv
from termcolor import colored
from rich.console import Console
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()
console = Console()

loader = TextLoader(file_path="./TextSplitters/docs/andromeda_paradox.txt")

docs = loader.load()
page = docs[0].page_content
print(colored("\nPage Content:", "magenta"))
print(colored(page, "yellow"))

print(colored("\nPage Metadata:", "magenta"))
console.print_json(data=docs[0].metadata)

splits = {}

char_splitter = CharacterTextSplitter(separator="", chunk_size=1000, chunk_overlap=2)
char_texts = char_splitter.split_text(page)
splits["character"] = {"split[0]": char_texts[0], "no.of splits": len(char_texts)}

word_splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=2)
texts = word_splitter.split_text(page)
splits["word"] = {"split[0]": texts[0], "no.of splits": len(texts)}

para_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=2)
texts = para_splitter.split_text(page)
splits["paragraph"] = {"split[0]": texts[0], "no.of splits": len(texts)}

print(colored("\Splits:", "magenta"))
console.print_json(data=splits)
