from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

MODEL = "google_genai:gemini-3.5-flash-lite"

def main():
    model = init_chat_model(MODEL)
    response = model.invoke("In one sentence, what is LangChain?")
    print(response.text)

if __name__ == "__main__":
    main()