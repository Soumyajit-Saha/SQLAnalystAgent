from contextlib import asynccontextmanager
import os
from typing import AsyncIterator
from uuid import UUID, uuid4

from fastapi import FastAPI, status
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agents.sql_analyst import sql_agent_graph

load_dotenv()


class QuestionRequest(BaseModel):
    user_question: str = Field(min_length=1, max_length=4000)


class ConversationCreated(BaseModel):
    conversation_id: UUID


class AgentResponse(BaseModel):
    conversation_id: UUID
    final_answer: str
    generated_sql_query: str
    is_safe: str
    comments: str


def initial_state() -> dict[str, object]:
    return {
        "messages": [],
        "user_question": "",
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": "",
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    redis_url = os.environ["REDIS_URL"]
    ttl_minutes = os.getenv("REDIS_CHECKPOINT_TTL_MINUTES")
    saver_options: dict[str, object] = {}
    if ttl_minutes:
        saver_options["ttl"] = {
            "default_ttl": int(ttl_minutes),
            "refresh_on_read": True,
        }

    async with AsyncRedisSaver.from_conn_string(
        redis_url, **saver_options
    ) as checkpointer:
        await checkpointer.asetup()
        app.state.sql_analyst = sql_agent_graph.compile(checkpointer=checkpointer)
        yield


app = FastAPI(lifespan=lifespan)


@app.post(
    "/conversations",
    response_model=ConversationCreated,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation() -> ConversationCreated:
    return ConversationCreated(conversation_id=uuid4())


@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=AgentResponse,
)
async def ask_question(
    conversation_id: UUID,
    request: QuestionRequest,
) -> AgentResponse:
    config = {"configurable": {"thread_id": str(conversation_id)}}
    graph = app.state.sql_analyst
    checkpoint = await graph.aget_state(config)
    state = dict(checkpoint.values) if checkpoint.values else initial_state()
    state["messages"] = []
    state["user_question"] = request.user_question

    result = await graph.ainvoke(state, config)
    return AgentResponse(
        conversation_id=conversation_id,
        final_answer=result["final_answer"],
        generated_sql_query=result["generated_sql_query"],
        is_safe=result["is_safe"],
        comments=result["comments"],
    )
