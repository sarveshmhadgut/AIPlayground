# imports & setups
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable
from utils.config import get_llm
from utils.database import save_row
from langchain_core.prompts import ChatPromptTemplate

PARAMS_CONFIGS = yaml.safe_load(
    (Path(__file__).parent / "configs/params.yaml").read_text()
)
PROMPTS_CONFIGS = yaml.safe_load(
    (Path(__file__).parent / "configs/prompts.yaml").read_text()
)
load_dotenv()


model = get_llm(PARAMS_CONFIGS["llm"])


@traceable(name="title_generator")
def generate_title(thread_id, conversation_history):
    if not conversation_history:
        return "new conversation"

    prompt = ChatPromptTemplate(
        [
            ("system", PROMPTS_CONFIGS["generate_title"]["system_prompt"]),
            ("user", PROMPTS_CONFIGS["generate_title"]["user_prompt"]),
        ]
    )

    chain = prompt | model
    res = chain.invoke({"conversation_history": conversation_history})
    save_row(thread_id=thread_id, thread_name=res)
    return res
