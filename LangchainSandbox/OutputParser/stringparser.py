from dotenv import load_dotenv
from termcolor import colored
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

template1 = PromptTemplate(
    template="Tell me about {topic} in about 50 words.",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Summarize following text in 3 bullet point:\n{text}",
    input_variables=["text", "count"],
)


prompt1 = template1.invoke({"topic": "Sony WF-1000XM5"})
res1 = model.invoke(prompt1)
parsed_res1 = parser.invoke(res1)
print("\n", colored("Description:", "magenta", attrs=["bold"]), "\n", parsed_res1)

prompt2 = template2.invoke({"text": res1.content})
res2 = model.invoke(prompt2)
parsed_res2 = parser.invoke(res2)
print("\n", colored("Summary:", "magenta", attrs=["bold"]), "\n", parsed_res2)


chain = template1 | model | parser | template2 | model | parser

res = chain.invoke({"topic": "Sony WF-1000XM5"})
print("\n", colored("Chained", "magenta", attrs=["bold"]), "\n", res)
