import json
from termcolor import colored
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

parser = JsonOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

template = PromptTemplate(
    template="Tell me a joke about {topic} as {character} according to following instructions:{parser_instructions}",
    input_variables=["topic", "character"],
    partial_variables={"parser_instructions": parser.get_format_instructions()},
)

prompt = template.invoke(
    {"topic": "Hair Dryer", "character": "Goofy from Mickey Mouse"}
)
print("\n", colored("Prompt:", "blue", attrs=["bold"]), "\n", prompt)

res = model.invoke(prompt)
parsed_res = parser.invoke(res)
print(
    "\n",
    colored("Response:", "magenta", attrs=["bold"]),
    "\n",
    json.dumps(parsed_res, indent=4, ensure_ascii=False),
)


chain = template | model | parser
res = chain.invoke({"topic": "Hair Dryer", "character": "Goofy from Mickey Mouse"})
print(
    "\n",
    colored("Chained:", "magenta", attrs=["bold"]),
    "\n",
    json.dumps(res, indent=4, ensure_ascii=False),
)
