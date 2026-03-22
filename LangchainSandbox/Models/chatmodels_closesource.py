from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model_gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
res_gemini = model_gemini.invoke("Who won superbowl 2000?")
print("Gemini:\n", res_gemini.text)

# model_gpt = ChatOpenAI(model="gpt-4", temperature=1.2, max_completion_tokens=100)
# res_gpt = model_gpt.invoke("Who won superbowl 2000?")
# print("GPT:\n", res_gpt)

# model_antropic = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=1.2)
# res_antropic = model_antropic.invoke("Who won superbowl 2000?")
# print("Antropic:\n", res_antropic)
