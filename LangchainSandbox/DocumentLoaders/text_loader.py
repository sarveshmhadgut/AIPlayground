from dotenv import load_dotenv
from termcolor import colored
from rich.console import Console
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader

load_dotenv()
console = Console()

# loader
loader = TextLoader(file_path="DocumentLoaders/docs/news.txt", encoding="utf-8")
docs = loader.load()[0]


print(colored("\nNews Articles:\n", "magenta"))
news_list = docs.page_content.split("\n\n")
for i, news in enumerate(news_list):
    print(f"{i + 1}) ", colored(news, "yellow"), end="\n\n")

print(colored("\nMetadata:\n", "magenta"))
console.print_json(data=docs.metadata)

# chain
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.7)
parser = StrOutputParser()
prompt = PromptTemplate(
    template="Categorize the news articles: {news} in Technology/Sports/Finance/Politics categories in one word.",
    input_variables=["news"],
)

chain = prompt | model | parser

# classification
print("News Classification:")
for i, news in enumerate(news_list):
    res = chain.invoke({"news": news})
    print(f"{i + 1}) {colored(news, 'yellow')} -> {colored(res, 'green')}\n")
