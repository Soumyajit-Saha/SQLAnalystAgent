from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

def pick_llm(level: str):
    '''
    Returns the LLM instance to be used with the model
    '''
    if level.lower() == "low":
        llm = ChatOpenAI(model_name="gpt-5.6-luna", temperature=0)
    elif level.lower() == "medium":
        llm = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0)
    elif level.lower() == 'high':
        llm = ChatOpenAI(model_name='gpt-5.6-sol', temperature=0)
    else:
        raise ValueError(f"Unsupported level: {level}")

    return llm