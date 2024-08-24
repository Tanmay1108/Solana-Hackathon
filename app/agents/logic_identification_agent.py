# import openai
from llama_index.core import Document, VectorStoreIndex
import requests
import json
# from langchain_openai import AzureChatOpenAI, ChatOpenAI


# # Define Tool class
# class Tool:
#     def __init__(self, name, func):
#         self.name = name
#         self.func = func

# # Define your tools
# class DiscordBotTool:
#     def __init__(self, token):
#         self.token = token

#     def run(self, query):
#         return f"Executed DiscordBotTool with query: {query}"

# class Web3WalletTool:
#     def __init__(self, rpc_url):
#         self.rpc_url = rpc_url

#     def create_wallet(self, query):
#         return f"Created Web3 Wallet with query: {query}"

# class GoogleCalendarTool:
#     def __init__(self, creds_file):
#         self.creds_file = creds_file

#     def add_event(self, query):
#         return f"Added event to Google Calendar with query: {query}"

# class LogicIdentificationAgent:
#     def __init__(self, tools, index):
#         self.tools = tools
#         self.index = index

#     def interpret_input(self, query):
#         # Use OpenAI to interpret the query and identify the correct tool logic
#         response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=f"Identify the correct tool for the following query: {query}",
#             max_tokens=100
#         )
#         return response['choices'][0]['text'].strip()

#     def route_to_tool(self, logic, query):
#         # Match the identified logic to the appropriate tool and call it
#         for tool in self.tools:
#             if logic.lower() in tool.name.lower():
#                 result = tool.func(query)
#                 return result
#         return "No suitable tool found."

#     def respond(self, query):
#         # Step 1: Identify the tool logic
#         logic = self.interpret_input(query)
        
#         # Step 2: Route to the tool and execute it
#         result = self.route_to_tool(logic, query)
        
#         # Step 3: Output the final result with steps
#         response = {
#             "steps": [
#                 {"step": 1, "description": f"Interpreted the query to identify logic: {logic}"},
#                 {"step": 2, "description": f"Routed to the appropriate tool and executed it"},
#             ],
#             "final_output": result
#         }
#         return response

# # Initialize your tools
# discord_bot_tool = Tool(name="DiscordBot", func=DiscordBotTool(token='your-discord-token').run)
# web3_wallet_tool = Tool(name="Web3Wallet", func=Web3WalletTool(rpc_url='your-rpc-url').create_wallet)
# google_calendar_tool = Tool(name="GoogleCalendar", func=GoogleCalendarTool(creds_file='your-creds.json').add_event)

# tools = [discord_bot_tool, web3_wallet_tool, google_calendar_tool]

# # Create LlamaIndex for enhanced functionality
# documents = [
#     SimpleNode(content="DiscordBotTool handles Discord bot functionalities"),
#     SimpleNode(content="Web3WalletTool manages web3 wallet operations"),
#     SimpleNode(content="GoogleCalendarTool adds events to Google Calendar")
# ]

# index = GPTVectorStoreIndex(documents)

# # Create the agent
# agent = LogicIdentificationAgent(tools=tools, index=index)

# # Example API usage
# query = "show me my web3 wallet"
# response = agent.respond(query)
# print(response)





from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Mistral 7B model and tokenizer with MPS support
# model_name = "mistralai/Mistral-Nemo-Instruct-2407"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Move model to MPS (if available) or CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model = model.to(device)

class LogicIdentificationAgent:
    def __init__(self, tools, index):
        self.tools = tools
        self.index = index

    # def interpret_input(self, query):
    #     # Use Mistral 7B to interpret the query and identify the correct tool logic
    #     inputs = tokenizer(f"Identify the correct tool for the following query: {query}", return_tensors="pt").to(device)
    #     with torch.no_grad():
    #         outputs = model.generate(**inputs, max_length=100)
    #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return response.strip()

    def interpret_input(self, query):
        # Use local Ollama model via HTTP request to interpret the query and identify the correct tool logic
        # prompt = f"Identify the correct tool for the following query: {query}"
        tool_list = []
        for tool in self.tools:
            tool_dict = {}
            tool_dict['name'] = tool.name
            tool_dict['description'] = tool.desc
            tool_list.append(tool_dict)
        prompt = f"""
**System Prompt:**

"YOU ARE A SMART INTELLIGENT ASSISTANT TO THE USER. GIVEN THE USERS QUERY AND LIST OF AVAILABLE TOOLS and their functionality descriptions, PROVIDE A SUGGESTED SEQUENCE OF TOOL USAGE REQUIRED TO 
COMPLETE THE QUERY.

ASSUME THE USER NEEDS HELP WITH THIS TASK AND PROPOSE THE OPTIMAL SET OF ACTIONS TO TAKE USING ONLY THE PROVIDED TOOLS.

Return name of tool that is required for the query"

**User Prompt:**

"Tools:
{json.dumps(tool_list)}

Query: {query}

Provide name of the tool that is required to be used.
        """
        payload = {
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False,
            "temparature": 0,         
        }
        
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()  # Raise an error if the request failed
            
            result = response.json()
            return result['response'].strip()
        
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return "Error running the model"

    def route_to_tool(self, logic, query):
        # Match the identified logic to the appropriate tool and call it
        for tool in self.tools:
            if tool.name.lower() in logic.lower():
                result = tool.func(query)
                return result
        return "No suitable tool found."

    def respond(self, query):
        # Step 1: Identify the tool logic
        logic = self.interpret_input(query)
        
        # Step 2: Route to the tool and execute it
        result = self.route_to_tool(logic, query)
        
        # Step 3: Output the final result with steps
        response = {
            "steps": [
                {"step": 1, "description": f"Interpreted the query to identify logic: {logic}"},
                {"step": 2, "description": f"Routed to the appropriate tool and executed it"},
            ],
            "final_output": result
        }
        return response

# Define Tool class and tools (same as before)
class Tool:
    def __init__(self, name, func, desc):
        self.name = name
        self.func = func
        self.desc = desc

class DiscordBotTool:
    def __init__(self, token):
        self.token = token

    def run(self, query):
        return f"Executed DiscordBotTool with query: {query}"

class Web3WalletTool:
    def __init__(self, rpc_url):
        self.rpc_url = rpc_url

    def create_wallet(self, query):
        return f"Created Web3 Wallet with query: {query}"

class GoogleCalendarTool:
    def __init__(self, creds_file):
        self.creds_file = creds_file

    def add_event(self, query):
        return f"Added event to Google Calendar with query: {query}"

# Initialize your tools (same as before)
discord_bot_tool = Tool(name="DiscordBot", func=DiscordBotTool(token='your-discord-token').run, desc="A tool that is responsible for discord related tasks")
web3_wallet_tool = Tool(name="Web3Wallet", func=Web3WalletTool(rpc_url='your-rpc-url').create_wallet, desc="A tool that is responsbile for showing a web3 wallet")
google_calendar_tool = Tool(name="GoogleCalendar", func=GoogleCalendarTool(creds_file='your-creds.json').add_event, desc="A tool that is responsible for upadting google calendar")

tools = [discord_bot_tool, web3_wallet_tool, google_calendar_tool]

# Create LlamaIndex for enhanced functionality
# documents = [
#     Document(text="DiscordBotTool handles Discord bot functionalities"),
#     Document(text="Web3WalletTool manages web3 wallet operations"),
#     Document(text="GoogleCalendarTool adds events to Google Calendar")
# ]

# index = VectorStoreIndex.from_documents(documents, embed_model='local')

# Create the agent
agent = LogicIdentificationAgent(tools=tools, index=None)

# Example API usage
query = "show me my web3 wallet"
response = agent.respond(query)
print(response)
