# imports & setups
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable
from utils.config import get_llm
from utils.database import save_row
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

params_configs = yaml.safe_load(Path("configs/params.yaml").read_text())
prompt_configs = yaml.safe_load(Path("configs/prompts.yaml").read_text())
load_dotenv()


model = get_llm(params_configs["llm"])
parser = StrOutputParser()


@traceable(name="title_generator")
def generate_title(thread_id, conversation_history):
    if not conversation_history:
        return "new conversation"

    prompt = ChatPromptTemplate(
        [
            ("system", prompt_configs["generate_title"]["system_prompt"]),
            ("user", prompt_configs["generate_title"]["user_prompt"]),
        ]
    )

    chain = prompt | model | parser
    res = chain.invoke({"conversation_history": conversation_history})
    save_row(thread_id=thread_id, thread_name=res)
    return res
