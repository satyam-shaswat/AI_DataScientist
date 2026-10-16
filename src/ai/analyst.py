import json
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.ai.tools import AGGREGATIONS, call_tool

MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = """You are a data analyst assistant for an uploaded dataset.

Rules:
1. Never calculate, estimate or invent numbers. Every number in your answer must come from a tool result in this conversation.
2. Use the tools to answer. If you are unsure of column names, call get_dataset_summary first.
3. If a tool returns an error, read it and try again with corrected arguments, or tell the user plainly what went wrong.
4. If no tool can answer the question (for example questions about trained models, or about data that is not in the dataset), say so. Do not guess.
5. Correlation does not imply causation. Say so when you report one.
6. Keep answers short and state which columns you used."""


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "get_dataset_summary",
        "Overview of the dataset: rows, columns, column types, columns with missing values, duplicate rows.",
        {}, [],
    ),
    _schema(
        "get_column_statistics",
        "Statistics for one column: mean, median, quartiles for numerical columns, top values for categorical ones.",
        {"column": {"type": "string", "description": "Exact column name"}},
        ["column"],
    ),
    _schema(
        "group_by_column",
        "Compare groups. If value_column is numerical, aggregate it per group. "
        "If value_column is categorical (like churn), give the percent of each value within each group.",
        {
            "group_column": {"type": "string", "description": "Column that defines the groups"},
            "value_column": {"type": "string", "description": "Column to summarise per group"},
            "aggregation": {"type": "string", "enum": AGGREGATIONS, "description": "Used only for numerical value columns"},
        },
        ["group_column", "value_column"],
    ),
    _schema(
        "calculate_correlation",
        "Pearson correlation between two numerical columns.",
        {
            "column1": {"type": "string", "description": "First numerical column"},
            "column2": {"type": "string", "description": "Second numerical column"},
        },
        ["column1", "column2"],
    ),
    _schema(
        "detect_outliers",
        "Potential outliers in a numerical column using the IQR method.",
        {"column": {"type": "string", "description": "Numerical column name"}},
        ["column"],
    ),
]


def build_llm():
    """Create the chat model named in the .env file."""
    load_dotenv()
    model = os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("LLM_MODEL is not set. Add it to your .env file (see .env.example).")
    return init_chat_model(model)


def _text(response):
    """The text of an AI reply (content can be a string or a list of blocks)."""
    content = response.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def chat_history(chat, max_messages=6):
    """Turn the last few stored chat turns into LangChain messages."""
    messages = []
    for item in chat[-max_messages:]:
        message_class = HumanMessage if item["role"] == "user" else AIMessage
        messages.append(message_class(content=item["content"]))
    return messages


def ask_analyst(df, question, llm, history=None, max_rounds=MAX_TOOL_ROUNDS):
    """Answer a question using the tool-calling loop.

    Returns {"answer": text, "trace": [every tool call with its result]}.
    """
    llm_with_tools = llm.bind_tools(TOOL_SCHEMAS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages += history or []
    messages.append(HumanMessage(content=question))

    trace = []
    for _ in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return {"answer": _text(response), "trace": trace}

        for call in response.tool_calls:
            result = call_tool(df, call["name"], call["args"])
            trace.append({"tool": call["name"], "arguments": call["args"], "result": result})
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))

    return {
        "answer": "I could not finish this analysis within the allowed number of tool calls.",
        "trace": trace,
    }