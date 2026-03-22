## Depricated method
from dotenv import load_dotenv
from termcolor import colored
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

llm_schema = [
    ResponseSchema(name="Comedian", description="Name of the comedian"),
    ResponseSchema(name="Setup", descption="Setup of the joke"),
    ResponseSchema(name="Punchline", descption="Punchline of the joke"),
]

parser = StructuredOutputParser.from_response_schema(llm_schema)

template = PromptTemplate(
    template="Tell me a joke about {topic} as {character} according to following instructions:{parser_instructions}",
    input_variables=["topic", "character"],
    partial_variables={"parser_instructions": parser.get_format_instructions()},
)

prompt = template.invoke(
    {"topic": "Paper Clip", "character": "Goofy from Mickey Mouse"}
)
print("\n", colored("Prompt:", "magenta", attrs=["bold"]), "\n", prompt)
res = model.invoke(prompt)

parsed_res = parser.invoke(res)
print("\n", colored("Response:", "magenta", attrs=["bold"]), "\n", parsed_res)

chain = template | model | parser

res = chain.invoke({"topic": "Paper Clip", "character": "Goofy from Mickey Mouse"})
print("\n", colored("Chained:", "magenta", attrs=["bold"]), "\n", res)
