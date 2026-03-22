from dotenv import load_dotenv
from typing import Dict
from rich.console import Console
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda

load_dotenv()
console = Console()

# string parser
parser = StrOutputParser()

# chain 1
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.7)

template = PromptTemplate(
    template="Explain the concept of {topic} using Counter-Strike gameplay analogies, mechanics, or roles, in approx 50 words.",
    input_variables=["topic"],
)
chain = RunnableSequence(template, model, parser)


# chain 2
def to_dict(text: str) -> Dict[str, int]:
    return {"text": text, "unique words": len(set(text.split(" ")))}


to_dict_runnable = RunnableLambda(to_dict)


# final chain
final_chain = RunnableSequence(chain, to_dict_runnable)

with console.status(
    "[bold magenta]Running sequential chains...[/bold magenta]",
    spinner="dots",
):
    res = final_chain.invoke({"topic": "Andromeda Paradox"})
console.print_json(data=res, indent=4)
