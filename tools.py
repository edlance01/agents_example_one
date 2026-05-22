from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime

# 1. Instantiate the underlying utilities
ddg_search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)


# 2. Wrap them using the @tool decorator (Fixed syntax)
@tool("DuckDuckGo_Search")  # <-- Just pass the string directly, no name=
def search_tool(query: str) -> str:
    """Searches the web using DuckDuckGo and returns a summary of the results."""
    return ddg_search.run(query)


@tool("Wikipedia_Search")  # <-- Just pass the string directly, no name=
def wiki_tool(query: str) -> str:
    """Use this tool to search Wikipedia for factual summaries of historical events, people, and concepts."""
    return wikipedia.run(query)

@tool
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """
      Saves or appends the provided research data/text to a local text file with a timestamp.
      Use this tool whenever you need to persist research summaries or results to disk.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"

# 3. Export the list for your main.py
my_agent_tools = [search_tool, wiki_tool, save_to_txt]
