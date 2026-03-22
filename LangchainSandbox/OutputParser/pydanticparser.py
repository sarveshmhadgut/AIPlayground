from dotenv import load_dotenv
from rich.console import Console
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

console = Console()

model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=1.5)


class LlmSchema(BaseModel):
    comedian: str = Field(description="Name of the comedian")
    setup: str = Field(description="Setup of the joke")
    punchline: str = Field(description="Punchline of the joke")


parser = PydanticOutputParser(pydantic_object=LlmSchema)

template = PromptTemplate(
    template=("Tell me a joke about {topic} as {character}.\n{parser_instructions}"),
    input_variables=["topic", "character"],
    partial_variables={"parser_instructions": parser.get_format_instructions()},
)


prompt = template.invoke(
    {"topic": "Sticky Note", "character": "Goofy from Mickey Mouse"}
)

console.print(prompt)

with console.status(
    "[bold magenta]Generating model response...[/bold magenta]",
    spinner="dots",
):
    res = model.invoke(prompt)

parsed_res = parser.invoke(res)

console.print("\n[bold magenta]Response:[/bold magenta]\n")
console.print_json(data=parsed_res.model_dump(), indent=4)


chain = template | model | parser

with console.status(
    "[bold magenta]Generating chained response...[/bold magenta]",
    spinner="dots",
):
    chained_res = chain.invoke(
        {"topic": "Sticky Note", "character": "Goofy from Mickey Mouse"}
    )

console.print("\n[bold magenta]Chained:[/bold magenta]\n")
console.print_json(data=chained_res.model_dump(), indent=4)
