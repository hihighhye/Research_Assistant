import streamlit as st
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
import json
import openai
import os
import datetime


def get_messages(thread_id):
    messages = openai.beta.threads.messages.list(
        thread_id=thread_id
    )
    messages = list(messages)
    messages.reverse()
  
    return messages

def send_message(thread_id, content):
    return openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )

def run_thread(assistant_id, thread_id):
    return openai.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
    
def get_run(run_id, thread_id):
    return openai.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id
    )

def get_tool_outputs(run_id, thread_id, status):
    run = get_run(run_id, thread_id)
    outputs = []

    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        
        status.update(label=f"Function '{function.name}' is running...", state="running")
        outputs.append({
            "output": functions_map[function.name](json.loads(function.arguments)),
            "tool_call_id": action_id,
        })

    return outputs

def submit_tool_outputs(outputs):
    return openai.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id =thread_id,
        tool_outputs=outputs,
    )

@st.cache_resource()
def create_assistant():
    assistant = openai.beta.assistants.create(
                    name="Streamlit - Research Assistant",
                    instructions="""
                        You are a senior research engineer.
                    
                        You research about given topic using Wikipedia or DuckDuckGo.
                        Once you find the website about the topic in DuckDuckGo, 
                        extract content from the website.
                        
                        After full research, summarize them as good to read.
                    """,
                    model="gpt-4o-mini",
                    tools=functions,
                )
    return assistant

def create_thread(query):
    thread = openai.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )
    return thread

def search_ddg(inputs):
    ddg = DuckDuckGoSearchResults()
    return ddg.run(inputs["query"])

def search_wiki(inputs):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(inputs["query"])

def load_website(inputs):
    loader = WebBaseLoader(inputs["url"])
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

def save_results(inputs):
    today = datetime.datetime.now().microsecond
    filename = f"{today}.txt"
    content = inputs["content"]
    with open(os.path.join("files", "agent", filename), "w", encoding="utf-8") as f:
        f.write(content)
    return json.dumps({"filename": filename, "content": content})

def extract_filename(inputs):
    return inputs["filename"]

def print_response(messages):
    with st.container():     
        content = messages[-1].content[0].text.value
        st.write(content.replace("$", "\$"))


functions_map = {
    "search_ddg": search_ddg,
    "search_wiki": search_wiki,
    "load_website": load_website,
    "save_results": save_results,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "search_ddg",
            "description": "This function returns the research result about given query from DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User's question"
                    }
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": "This function returns the research result about given query from Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User's question"
                    }
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_website",
            "description": "This function extracts content from given website.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The website url"
                    }
                },
                "required": ["url"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_results",
            "description": "This function saves the research result as a txt file \
                and returns the filename and the content as a json format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content, which is the well-organized summary of the research result."
                    },
                },
                "required": ["content"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_filename",
            "description": "This function extracts filename in given stringified json object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The saved filename"
                    },
                    "content": {
                        "type": "string",
                        "description": "The saved content"
                    },
                },
                "required": ["filename", "content"],
            }
        }
    },
]



st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍"
)

with st.sidebar:
    with st.form("OpenAI API Key Setting"):
        user_openai_api_key = st.text_input("Enter your OpenAI API key.")
        submitted = st.form_submit_button("Set")
        if submitted:
            os.environ['OPENAI_API_KEY'] = user_openai_api_key

            if "assistant_id" not in st.session_state:
                assistant = create_assistant()
                st.session_state["assistant_id"] = assistant.id

st.title("Research Assistant")

st.markdown(
    """
    ---

    Ask anything that you want to research,

    then our research assistant will find the information
    and return the result as a text file.
    """
)

st.divider()

query = st.chat_input("Ask anything that you want to research.")

if query:
    with st.chat_message("human"):
        st.markdown(query)

    with st.status("Preparing to run...") as prog:
        assistant_id = st.session_state["assistant_id"]

        thread = create_thread(query)
        thread_id = thread.id

        run = run_thread(assistant_id, thread_id)
        run_id = run.id

    with st.status("Calling functions...") as status:
        while (True):   
            thread_status = get_run(run_id, thread_id).status
            if thread_status == "completed":
                status.update(label="Completed", state="complete")
                break
            else:
                if thread_status == "requires_action":
                    outputs = get_tool_outputs(run_id, thread_id, status)
                    run = submit_tool_outputs(outputs)
    
    messages = get_messages(thread_id)
        
    with st.chat_message("assistant"):
        st.markdown("Here is the research result.")
        st.markdown(messages[-1].content[0].text.value)

               