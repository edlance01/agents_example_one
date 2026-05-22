from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

#
# NOTE: Agent api updated from langchain.  See modern_langchain_agent.txt
#
load_dotenv()

class ResearchResponse(BaseModel): # make sure you inherit from BaseModel to make it a pydantic model
    # you can define the structure of the response you want from the model and make it as complex as you want
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# llm = ChatOpenAI(model="gpt-4o-mini")
llm2 = ChatAnthropic(model="claude-haiku-4-5")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [(
    "system",
    """
     You are a research assistant that will help generate a research paper.
     Wrap the output is format and provide not other text \n{format_instructions}
     """,
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions()) # users parser and ResearchResponse class defined above to generate format instructions for the model


agent = create_tool_calling_agent(
    llm=llm2,
    prompt = prompt,
    tools = []
)

# test the agent with a query
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "What is the current state of research on quantum computing?"})
# response = agent_executor.run(query="What is the current state of research on quantum computing?")
# print(response)
