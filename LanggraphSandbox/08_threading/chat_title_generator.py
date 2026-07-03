# imports & setups
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

PARAMS_CONFIGS = yaml.safe_load((Path(__file__).parent / "configs/params.yaml").read_text())
PROMPTS_CONFIGS = yaml.safe_load((Path(__file__).parent / "configs/prompts.yaml").read_text())
load_dotenv()

# model & parser
model = ChatGoogleGenerativeAI(**PARAMS_CONFIGS["llm"])
parser = StrOutputParser()


def generate_title(conversation_history):
    if not conversation_history:
        return "new conversation"

    prompt = ChatPromptTemplate(
        [
            ("system", PROMPTS_CONFIGS["generate_title"]),
            ("user", "Conversation:\n\t{conversation_history}"),
        ]
    )

    chain = prompt | model | parser
    res = chain.invoke({"conversation_history": conversation_history})
    return res
