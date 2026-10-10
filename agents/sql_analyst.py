from utils.llm_pick import pick_llm
from Models.schema import AgentSchema
from langchain_core.messages import HumanMessage
from utils.database import DatabaseUtils
from dotenv import load_dotenv
import os

load_dotenv()
def curate_question(state: AgentSchema) -> AgentSchema:
    user_question = state.user_question
    llm = pick_llm('low')
    response = llm.invoke(f"Curate the following question: {user_question}")

    state.curated_ques =  response
    state.messages += [HumanMessage(content=f"{response}")]
    return state


def prompt_query_context(state: AgentSchema) -> AgentSchema:
    curated_question = state.curated_ques
    db_obj = DatabaseUtils({
        'host': os.getenv('host'),
        'port': int(os.getenv('port')),
        'user': os.getenv('user'),
        'password': os.getenv('password'),
        'dbname': os.getenv('dbname')
    })

    schema_info = db_obj.schema_details("public")

    prompt = f"""
    You are an SQL analyst agent. Your task is to convert the user's natural language 
    query into Postgres SQL query that can be executed on the database. You are provided 
    with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that 
    you can understand the structure of the database and generate an accurate SQL query.
    Unless user explicitly asks for specific number of rows, always limit the output to 10 rows.
    Note - Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications.  
    
    User's Original Query: {curated_question}

    Database Schema Details:
    {schema_info}
    
    """    

    state.prompt_query_context = prompt

    return state


# Generate SQL Query Node
def generate_sql(state: AgentSchema) -> AgentSchema:

    prompt = state.prompt_query_context

    llm = pick_llm("medium")  # Pick the appropriate LLM based on the level of the question

    generated_sql_query = llm.invoke(prompt).content  # Generate the SQL query using the LLM

    state.generated_sql_query = generated_sql_query

    return state

