from langchain.tools import tool
from src.config.settings import settings
from tavily import TavilyClient

tavily_client = TavilyClient(api_key = settings.tavily_api_key)

@tool("Web_Search_Tool")
def web_search_tool(
    query: str, 
    max_results: int = 1,
):
    """
    Perform an advanced web search using the Tavily API to find real-time information.
    Args:
        query (str): The search query. Be specific (e.g., "3 day itinerary for Tokyo" instead of "Tokyo").
        max_results (int): The maximum number of search results to return. Defaults to 5.
    Returns:
        list[dict]: A list of dictionaries containing 'url', 'content', and 'title' for each result.
    """
    response = tavily_client.search(
                query=query,
                search_depth="advanced",
                topic="general",
                max_results=max_results
            )
    response = response['results'][0]['content']
    return response