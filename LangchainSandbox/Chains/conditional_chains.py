from typing import Literal
from dotenv import load_dotenv
from rich.console import Console
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()
console = Console()
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.7)

news = """
Big Tech stocks tumble amid AI spending pledges despite strong quarterly earnings
"""


# news labelling parser schema
class LabelSchema(BaseModel):
    label: Literal["Finance", "Politics", "Technology"] = Field(
        description="Category label of the news headline"
    )
    headline: str = Field(description="Original headline of the news as it is")


# news labelling parser
label_parser = PydanticOutputParser(pydantic_object=LabelSchema)

# news labelling template
label_template = PromptTemplate(
    template="Label the news headline '{news}' according to parser instruction:\n{parser_instructions}",
    input_variables=["news"],
    partial_variables={"parser_instructions": label_parser.get_format_instructions()},
)


# news labelling chain
label_chain = label_template | model | label_parser


# runnable -> dictinary
def as_dict(runnable) -> dict:
    return runnable.model_dump()


as_dict_runnable = RunnableLambda(as_dict)


# mail drafting parser schema
class MailSchema(BaseModel):
    subject: str = Field(description="Subject of the mail")
    body: str = Field(description="Body of the mail in about 50 words")


# mail drafting parser
mail_parser = PydanticOutputParser(pydantic_object=MailSchema)

# financial mail drafting template
finance_mail_template = PromptTemplate(
    template=(
        "Draft an email to the company's finance advisor asking them to review the following news headline and analyze its implications for the company.\n"
        "Headline:\n{headline}\n"
        "{parser_instructions}"
    ),
    input_variables=["headline"],
    partial_variables={"parser_instructions": mail_parser.get_format_instructions()},
)

# tech mail drafting template
tech_mail_template = PromptTemplate(
    template=(
        "Draft an email to the company's head of engineering requesting them to review the following news headline and assess its technical and operational implications for the company.\n"
        "Headline:\n{headline}\n"
        "{parser_instructions}"
    ),
    input_variables=["headline"],
    partial_variables={"parser_instructions": mail_parser.get_format_instructions()},
)

# finance chain
finance_chain = finance_mail_template | model | mail_parser
# tech chain
tech_chain = tech_mail_template | model | mail_parser

# conditional runnable
conditional_runnable = RunnableBranch(
    (lambda x: x["label"] == "Finance", finance_chain),
    tech_chain,
)

final_chain = label_chain | as_dict_runnable | conditional_runnable

with console.status(
    "[bold magenta]Running conditional chains...[/bold magenta]",
    spinner="dots",
):
    res = final_chain.invoke({"news": news})

console.print_json(data=as_dict(res), indent=4)
