import os
from typing import Annotated
import yaml
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
from autogen import UserProxyAgent, AssistantAgent, ConversableAgent, register_function
import autogen
from autogen.agentchat.contrib.capabilities.teachability import Teachability


load_dotenv()
scraping_api_key = os.getenv("SCRAPING_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")
config_list = [
    {
    "model": os.getenv("OPENAI_MODEL"),
    "api_key": os.getenv("OPENAI_API_KEY")
    }
]
airtable_api_key = os.getenv("AIRTABLE_API_KEY")

# model gpt-4-turbo-preview gpt-3.5-turbo-16k-0613

# import Agents Prompt from yaml file

with open('prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)

chef_description = prompts['chef']['description']
chef_prompt = prompts['chef']['prompt']
publisher_description = prompts['publisher']['description']
publisher_prompt = prompts['publisher']['prompt']
publisher_manager_description = prompts['publisher_manager']['description']
publisher_manager_prompt = prompts['publisher_manager']['prompt']



# ---------- CREATE A FUNCTION ------------- #

# Function for google search
def google_search(
        search_keyword: Annotated[str, "Optimal search keywords are those most likely to yield relevant results for the information you seek."]
        ) -> Annotated[str, "Response from  google search"]:
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": search_keyword
    })

    headers = {
        'X-API-KEY': serp_api_key, 
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print("RESPONSE:", response.text)
    return response.text

# Function for scraping
def summary(objective, content):
    llm = ChatOpenAI(temperature =0, model = "gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose= True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)
    
    return output

def web_scraping(
        objective: Annotated[str, "the goal of scraping the website. e.g. any specific type of informtion you are looking for?"], 
        url: Annotated[str, "the url website you want to scrape"]
        ) -> Annotated[requests.Response, "to send the post request"]:
    # scrape website, and also will summarize the content based on objective
    # objective is the original objective & task that user give to the agent

    print(f"Scraping website...{url}\n\nAnd this is the objective {objective}\n\n")


    # define the headers for the request
    headers= {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json'
    }

    payload = {
        "api_key": scraping_api_key,
        "url": url,
        "headers": json.dumps(headers),
    }

    # define the data to be sent in the request
    data = {
        "url": url
    }

    # convert Python object to JSON string
    data_json =json.dumps(data)

    # send the POST request
    response = requests.post(f"https://scraping.narf.ai/api/v1/", params=payload) 

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTT:", text)
        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}") 

 

# ---------- CREATE AGENT ------------- #

# Create user proxy
user_proxy = UserProxyAgent(name="user_proxy", description="User Proxy Agent",
     is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
     human_input_mode="ALWAYS", #  use "ALWAYS" to interact with the chat, viceversa use "NEVER"
     max_consecutive_auto_reply=3,
     )

# Create Memory Agent
chef = ConversableAgent(
    name = "Chef",
    system_message=chef_prompt,
    description=chef_description, 
    llm_config={
        "config_list":
        [
            {
                "model": "gpt-4-turbo-preview", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ],
        "timeout":120
    }
)

# Create Publisher
publisher = AssistantAgent(
    name="publisher",
    system_message=publisher_prompt,
    description=publisher_description, 
    llm_config={
        "config_list":
        [
            {
                "model": "gpt-4-turbo-preview", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ],
        "timeout":120
    }
)

# Create Publisher_Manager
publisher_manager = AssistantAgent(
    name="publisher_manager",
    system_message=publisher_manager_prompt,
    description=publisher_manager_description, 
    llm_config={
        "config_list":
        [
            {
                "model": "gpt-4-turbo-preview", 
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ],
        "timeout":120
    }
)

# Istantiate a Teachability object. Its parameters are all optional.
teachability = Teachability(
    reset_db=True, # Use True to force-reset the memo DB, and False ro use an exiating knowledge database
    path_to_db_dir="./tmp/teachability_db",
)
# Now add teachability to the agent.
teachability.add_to_agent(chef)

# register_function(
#     google_search,
#     caller=chase,
#     executor=chase,
#     name="google_search",
#     description="Google Search to return results of research keywords"
# )

# register_function(
#     web_scraping,
#     caller=chase,
#     executor=chase,
#     name="web_scraping",
#     description="scrape website content based on url"
# )


# create the Speaker selectipon method:
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    if last_speaker is user_proxy:
       if "ok" in messages[-1]["content"]:
           return publisher
       # init -> retrieve (director)
       return chef
    elif last_speaker is chef:
       return user_proxy
    elif last_speaker is publisher:
       return publisher_manager
    elif last_speaker is publisher_manager:
        return None


# Create group chat
groupchat = autogen.GroupChat(
    agents= [chef, publisher, publisher_manager, user_proxy], 
    messages=[], 
    max_round=12, 
    speaker_selection_method=state_transition) # use "auto" to exploit LLM to automatically select the next speeker or "round_robin" to follow a cascade flow

group_chat_manager = autogen.GroupChatManager(
    groupchat=groupchat, llm_config={
        "config_list":config_list})

# ----------- START CONVERSATION --------- #
user_proxy.initiate_chat(group_chat_manager)
# print(researcher.llm_config["tools"])"