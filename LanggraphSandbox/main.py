from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7,
    max_output_tokens=2000,
)


def main():
    print(model.invoke("hi"))


if __name__ == "__main__":
    main()
