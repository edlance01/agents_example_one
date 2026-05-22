from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from tools import my_agent_tools

load_dotenv()


# 1. Define the desired output structure
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# 2. Setup LLM and tools
llm2 = ChatAnthropic(
    model="claude-haiku-4-5"
)  # Using current recommended Claude model
tools = my_agent_tools

# 3. Simplify the Agent Prompt (Remove formatting instructions from here!)
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research assistant. Use your tools to find accurate, up-to-date information to answer the user's request comprehensively.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# 4. Build the Agent Executor
agent = create_tool_calling_agent(llm=llm2, prompt=agent_prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Run the Agent
query = input("What can I help you research today? \n> ")
print("\n--- Agent is researching... ---")
raw_response = agent_executor.invoke({"query": query})

# Extract the text answer the agent generated after using its tools
agent_final_text = raw_response["output"]

# 6. Now, force that final answer into your Pydantic structure
print("\n--- Formatting results into Structured JSON... ---")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

formatting_prompt = ChatPromptTemplate.from_template(
    "You are a data formatting assistant.\n"
    "Take the following research text and format it strictly according to the structure below.\n"
    "{format_instructions}\n\n"
    "Research Text:\n{text}"
)

# Chain the LLM directly to the parser for the final cleanup
formatting_chain = (
    formatting_prompt.partial(format_instructions=parser.get_format_instructions())
    | llm2
    | parser
)

try:
    structured_response = formatting_chain.invoke({"text": agent_final_text})

    print("\n--- SUCCESS! ---")
    print(f"Topic: {structured_response.topic}")
    print(f"Summary: {structured_response.summary}")
    print(f"Sources: {structured_response.sources}")
    print(f"Tools Used: {structured_response.tools_used}")

except Exception as e:
    print(f"\nError parsing response: {e}")
    print(f"Raw text was: {agent_final_text}")
