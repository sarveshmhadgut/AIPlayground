from dotenv import load_dotenv
from rich.console import Console
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()
console = Console()
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.5)

news = """
Gold prices climbed back above $5,000 per ounce amid renewed buying interest after recent volatility, signalling possible safe-haven demand
"""


# article generation parser schema
class ArticleSchema(BaseModel):
    article: str = Field(description="Content of the article in about 100 words")


# article generation parser
article_parser = PydanticOutputParser(pydantic_object=ArticleSchema)


# article generation template
article_template = PromptTemplate(
    template="Generate a spicy news article for headline: '{headline}' and according to following instructions:\n{parser_instructions}.",
    input_variables=["headline"],
    partial_variables={"parser_instructions": article_parser.get_format_instructions()},
)


# article generation chain
article_chain = article_template | model | article_parser


# runnable -> dictionary
def as_dict(runnable):
    return runnable.model_dump()


as_dict_runnable = RunnableLambda(as_dict)


# blog generation parser schema
class BlogSchema(BaseModel):
    title: str = Field(description="Title of the blogpost")
    content: str = Field(description="Content of the blogpost in about 50 words")


# blog generation parser
blog_parser = PydanticOutputParser(pydantic_object=BlogSchema)

# goofy's blog generation parser
goofy_template = PromptTemplate(
    template="Draft content for goofy's blogpost for {article}. according to following instructions:\n{parser_instructions}.",
    input_variables=["article"],
    partial_variables={"parser_instructions": blog_parser.get_format_instructions()},
)

# donald duck's blog generation parser
donald_template = PromptTemplate(
    template="Draft content for Donald duck's blogpost for {article}. according to following instructions:\n{parser_instructions}.",
    input_variables=["article"],
    partial_variables={"parser_instructions": blog_parser.get_format_instructions()},
)

# goofy's blog generation chain
goofy_chain = goofy_template | model | blog_parser | as_dict_runnable
# donald duck's blog generation chain
donald_chain = donald_template | model | blog_parser | as_dict_runnable

final_chain = (
    article_chain
    | as_dict_runnable
    | RunnableParallel(branches={"Goofy": goofy_chain, "Scooby": donald_chain})
)

with console.status(
    "[bold magenta]Running parallel chains...[/bold magenta]",
    spinner="dots",
):
    res = final_chain.invoke({"headline": news})
console.print_json(data=res, indent=4)
