# import json
# from dotenv import load_dotenv
# from rich.console import Console
# from rich.status import Status
# from rich.pretty import Pretty
# from pydantic import BaseModel, Field
# from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.output_parsers import PydanticOutputParser

# load_dotenv()

# console = Console()

# model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=1.5)


# class LlmSchema(BaseModel):
#     comedian: str = Field(description="Name of the comedian")
#     setup: str = Field(description="Setup of the joke")
#     punchline: str = Field(description="Punchline of the joke")


# parser = PydanticOutputParser(pydantic_object=LlmSchema)

# template = PromptTemplate(
#     template=("Tell me a joke about {topic} as {character}.\n{parser_instructions}"),
#     input_variables=["topic", "character"],
#     partial_variables={"parser_instructions": parser.get_format_instructions()},
# )

# # ---------- Single invoke ----------

# prompt = template.invoke(
#     {"topic": "Sticky Note", "character": "Goofy from Mickey Mouse"}
# )

# # console.print("\n[bold magenta]Prompt:[/bold magenta]\n")
# console.print(prompt)

# with console.status(
#     "[bold magenta]Generating model response...[/bold magenta]",
#     spinner="dots",
# ):
#     res = model.invoke(prompt)

# parsed_res = parser.invoke(res)

# console.print("\n[bold magenta]Response:[/bold magenta]\n")
# console.print(Pretty(parsed_res.model_dump()))


# chain = template | model | parser

# with console.status(
#     "[bold magenta]Generating chained response...[/bold magenta]",
#     spinner="dots",
# ):
#     chained_res = chain.invoke(
#         {"topic": "Sticky Note", "character": "Goofy from Mickey Mouse"}
#     )

# console.print("\n[bold magenta]Chained:[/bold magenta]\n")
# console.print(Pretty(chained_res.model_dump()))
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme(
    {
        "json.key": "#C1A2FF",
        "json.string": "#C1A2FF",
        "json.escape": "#C1A2FF",
        "json.number": "#C1A2FF",
        "json.boolean": "#C1A2FF",
        "json.null": "#C1A2FF",
        "json.brace": "#C1A2FF",
        "json.colon": "#C1A2FF",
        "json.comma": "#C1A2FF",
    }
)

console = Console(
    theme=custom_theme,
    force_terminal=True,
    color_system="truecolor",
)

console.print_json(data={"test": "value"}, indent=4)
