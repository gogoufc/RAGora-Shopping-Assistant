#chat_app.py
import uuid
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver 
from agent_graph import workflow 
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Load environment variables (API keys)
load_dotenv()

def run_chat():
    # Initialize rich console
    console = Console()

    # 1. SETUP MEMORY
    # MemorySaver keeps the conversation history in RAM during the session
    memory = MemorySaver()
    
    # 2. COMPILE GRAPH
    # We re-compile the workflow here to inject the Checkpointer (Memory)
    local_app = workflow.compile(checkpointer=memory)

    # 3. SESSION SETUP
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # WELCOME BANNER (Updated with your style)
    welcome_text = Text("RAGora: The Smart Shopping Pal :)", justify="center", style="bold white")
    console.print(Panel(welcome_text, style="yellow", subtitle=f"Session ID: {thread_id[:8]}"))
    
    # ΟΔΗΓΙΕΣ ΜΕ ΚΟΚΚΙΝΟ (RED) ΓΙΑ ΤΟ QUIT/EXIT
    console.print("[dim]Type '[bold red]quit[/bold red]' or '[bold red]exit[/bold red]' to end the chat.[/dim]\n")

    # 4. CHAT LOOP
    while True:
        try:
            # Get User Input
            user_input = console.input("[bold green]You:[/bold green] ")
            
            if user_input.lower() in ["quit", "exit"]:
                console.print("\n[bold yellow]Goodbye! Happy Shopping![/bold yellow]")
                break
            
            if not user_input.strip():
                continue
                
            input_message = HumanMessage(content=user_input)
            
            # Show a "Thinking" indicator
            with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
                
                # Run the Agent
                events = local_app.stream(
                    {"messages": [input_message]}, 
                    config=config, 
                    stream_mode="values"
                )
                
                final_snapshot = None
                for event in events:
                    final_snapshot = event 

            # Process the Final Response
            if final_snapshot and "messages" in final_snapshot:
                messages = final_snapshot["messages"]
                
                if messages:
                    last_msg = messages[-1]
                    
                    if isinstance(last_msg, AIMessage):
                        # Use a Rule line for cleaner separation
                        console.rule("[bold cyan]Agent[/bold cyan]")
                        
                        # Render Markdown nicely
                        md = Markdown(last_msg.content)
                        console.print(md)
                        console.print() # Empty line for spacing
                        
                    elif isinstance(last_msg, BaseMessage) and last_msg.type != "human":
                         console.print(f"[bold red]System:[/bold red] {last_msg.content}")
            
        except Exception as e:
            console.print(f"\n[bold red]An error occurred:[/bold red] {e}")
            console.print("Please try again.\n")

if __name__ == "__main__":
    run_chat()