from dotenv import load_dotenv
from rich.console import Console
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
console = Console()

# string parser
parser = StrOutputParser()

# chain 1
model1 = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.7)

template1 = PromptTemplate(
    template="Explain the concept of {topic} using Counter-Strike analogies in approx 50 words.",
    input_variables=["topic"],
)
chain1 = RunnableSequence(template1, model1, parser)

# chain 2
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

template2 = PromptTemplate(
    template="Explain the concept of {topic} using basketball analogies in approx 50 words.",
    input_variables=["text"],
)
chain2 = RunnableSequence(template2, model2, parser)

# final chain
final_chain = RunnableParallel({"CS": chain1, "Basketball": chain2})

with console.status(
    "[bold magenta]Running parallel chains...[/bold magenta]",
    spinner="dots",
):
    res = final_chain.invoke({"topic": "Andromeda Paradox"})

console.print_json(data=res, indent=4)
