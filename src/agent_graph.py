#agent_graph.py
import os
import time
import operator
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END

# Import the search tool
from tools_c4 import search_products

# Load environment variables
load_dotenv()

# LLM Configuration
# Using GLM-4.5-Air via OpenRouter (Free Tier)
llm = ChatOpenAI(
    model="z-ai/glm-4.5-air:free",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    max_tokens=2000,
    request_timeout=60,
    max_retries=3
)

# Agent State definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str 

# NODE 1: SUPERVISOR (ROUTER)
def supervisor_node(state: AgentState):
    time.sleep(1) 
    messages = state.get('messages', [])
    if not messages:
        return {"next_step": "chit_chat"}

    last_message = messages[-1]
    if not last_message.content or not last_message.content.strip():
        return {"next_step": "chit_chat"}

    
    system_prompt = """
    You are a smart router for a shopping assistant. 
    Analyze the conversation history.
    
    ROUTING RULES:
    1. **SEARCH READY (Priority)**: Route to 'product_search' if:
       - The user provides specific product keywords.
       - **OR** the user describes a specific **Scenario** or **Trait** (e.g. "he likes car rides and jokes", "I fall asleep in the car").
       - **LOGIC**: Do NOT ask "what kind of product" if the user has already described the vibe/scenario. Take initiative and search for relevant items (e.g. car accessories) immediately.
       
    2. **CLARIFICATION NEEDED**: Route to 'chit_chat' ONLY if the request is empty of detail (e.g. just "gift for him" with zero context).
    
    3. **FOLLOW-UP**: If the user asks about previous results, route to 'product_search'.
    
    4. **GENERAL**: Route to 'chit_chat' for hello/thanks.
    
    Respond ONLY with one word: 'product_search' or 'chit_chat'.
    """
    
    classification_messages = [SystemMessage(content=system_prompt)] + messages 
    
    try:
        response = llm.invoke(classification_messages)
        decision = response.content.strip().lower()
        
        if "product" in decision:
            return {"next_step": "product_search"}
        return {"next_step": "chit_chat"}
        
    except Exception as e:
        print(f"Supervisor API Error: {e}")
        return {"next_step": "chit_chat"}

# NODE 2: PRODUCT SEARCH
def product_search_node(state: AgentState):
    time.sleep(1)
    messages = state.get('messages', [])
    
    recent_history = messages[-10:] if len(messages) > 4 else messages
    history_text = "\n".join([f"{m.type}: {m.content}" for m in recent_history])
    
    #Prompt: SCENARIO TO PRODUCT TRANSLATION
    query_gen_system = """
    You are a Search Query Generator.
    Your goal: Translate the user's description into a SINGLE, standalone search string.
    
    INSTRUCTIONS:
    - If the user describes a **Scenario** (e.g. "I fall asleep in the car"), convert it to relevant products.
      - "fall asleep in car, funny" -> "funny car travel pillow passenger gag gift"
      - "likes cooking, sci-fi" -> "star wars kitchen gadgets"
    
    - Do NOT ask questions. Output ONLY the search string.
    """
    
    try:
        query_response = llm.invoke([
            SystemMessage(content=query_gen_system),
            HumanMessage(content=f"Conversation History:\n{history_text}")
        ])
        refined_query = query_response.content.strip()
    except Exception:
        refined_query = messages[-1].content if messages else ""

    if not refined_query:
         return {"messages": [AIMessage(content="I didn't quite get that. Could you describe the product again?")]}

    try:
        products = search_products(refined_query, top_k=6)
    except Exception:
        products = []
    
    products_context = ""
    if products:
        for i, p in enumerate(products, 1):
            products_context += f"{i}. {p['title']}\n   Category: {p['category']}\n   Description: {p['description'][:300]}...\n\n"
    else:
        products_context = "No specific products found."

    system_instructions = """
    You are a helpful Shopping Assistant.
    IMPORTANT: No live prices available. Do NOT invent prices.
    
    STRICT FORMATTING RULES:
    1. Present the Top 3 options.
    2. Start each option with the **Product Name**.
    3. For each option, write a short bullet point explaining "Why it fits".
    
    If results are generic, pick the funniest/most relevant ones fitting the 'Scenario'.
    """

    user_content_for_llm = f"""
    USER'S LATEST REQUEST: "{messages[-1].content}"
    CONTEXT (REFINED SEARCH QUERY): "{refined_query}"
    SEARCH RESULTS:
    {products_context}
    Please provide your recommendation.
    """
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content=user_content_for_llm)
        ])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content="Connection error.")]}

# NODE 3: CHIT CHAT
def chit_chat_node(state: AgentState):
    time.sleep(1)
    messages = state.get('messages', [])
    last_message = messages[-1] if messages else SystemMessage(content="")

    history_text = "\n".join([f"{m.type}: {m.content}" for m in messages[-6:]])

    system_prompt = f"""
    You are a helpful Amazon Shopping Assistant.
    
    CONVERSATION HISTORY:
    {history_text}
    
    YOUR GOAL:
    1. If the user just said "Hello", ask what they are looking for.
    2. If the user gave a VERY vague request (e.g. "gift"), ask 2-3 clarifying questions (Interests, Style).
    3. **IMPORTANT**: If the user has already answered your questions with specific details, **do not ask more**. (The system should have routed them to search, but if you are here, apologize and suggest we try searching).
    
    Be polite and concise.
    """
    
    conversation = [SystemMessage(content=system_prompt), last_message]
        
    try:
        response = llm.invoke(conversation)
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content="Connection error.")]}

# GRAPH CONSTRUCTION
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("product_search", product_search_node)
workflow.add_node("chit_chat", chit_chat_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda state: state['next_step'], {"product_search": "product_search", "chit_chat": "chit_chat"})
workflow.add_edge("product_search", END)
workflow.add_edge("chit_chat", END)
app = workflow.compile()
print("Agent graph loaded successfully.")
#print("Generating graph image...")
#try:
#    image_data = app.get_graph().draw_mermaid_png()
#    with open("agent_graph_diagram.png", "wb") as f:
#        f.write(image_data)
#    print("Success! Image saved as agent_graph_diagram.png")
#except Exception as e:
#    print(f"Failed to generate image: {e}")