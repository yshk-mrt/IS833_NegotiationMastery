import streamlit as st
import openai
import os
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import ChatMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.globals import set_debug
from langchain.output_parsers import OutputFixingParser
from langchain.schema import OutputParserException
import random
from typing import Any, Dict, List, Union
from langchain.schema import AgentAction

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

#set_debug(True)
openai.api_key = os.environ.get('OPENAI_API_KEY')
#openai.api_key = "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"#os.environ.get('OPENAI_API_KEY')
#openai_api_key = "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"

azure_blob_connection_str = os.environ.get('AZURE_BLOB_CONNECTION_STR')

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token.replace("$", r"\$")
        self.container.markdown(self.text + "|")
    
    def on_llm_end(self, token: str, **kwargs) -> None:
        self.container.markdown(self.text)

class SalarySearchHandler(BaseCallbackHandler):
    def __init__(self, placeholder, initial_text="Thinking"):
        self.placeholder = placeholder
        self.text = initial_text
        self.counter = 0
        self.placeholder.markdown(self.text + "|")
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += "." if self.counter % 2 else ""
        self.placeholder.markdown(self.text + "|")
        self.counter += 1
        #st.chat_message("user").write(self.text)
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        self.text = "Searching for salary information"
        self.placeholder.markdown(self.text)
        #self.placeholder.write(f"on_tool_start {serialized['name']}")
    
    def on_llm_end(self, token: str, **kwargs) -> None:
        self.placeholder.empty()

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass
        #self.placeholder.write(f"Action: {action.tool}, Input:{action.tool_input}")

def load_llm(stream_handler):
    llm = ChatOpenAI(model='gpt-4', streaming=True, callbacks=[stream_handler])
    #llm = ChatOpenAI(model='gpt-4', streaming=True, callbacks=[stream_handler], openai_api_key=openai.api_key)
    return llm

def create_system_prompt(user_role, optional_instruction):
    # salary_multiplier = st.session_state.salary_multiplier
    # sign_on_bonus_ratio_to_base_salary = st.session_state.sign_on_bonus_ratio_to_base_salary
    # min_salary = st.session_state.min_salary
    # max_salary = st.session_state.max_salary
    # average_salary = st.session_state.average_salary

    #format_instructions = output_parser.get_format_instructions()

    role = "I want to do a role-playing exercise and I will be a police hostage negotiator. I will be the hostage negotiator. You will be the criminal. You are driven by greed. You do not want to hurt any of the hostages."
    task = "You will assume the role of the criminal. And wait for me to contact your to begin the negotiations. You will not act as the police negotiator at any time."#You will start by pretending to be a junior police officer and approach me to tell me the criminal has been reached by phone, and you want the negotiator's response. You will then ask what I want to say next. You will then wait for me to respond;
    goal = "To reach a deal with the officer. You value money first, freedom second."
    user_role = "Police Negotiator"
    condition = f"The amount of money, the number of hostages, and the location of the incident are all up to you to decide unless the user defines them."
    rule = "Only act as the criminal or the users police assistant. Do not play the role of the lead police negotiator that will be played by the user."
    #optional_instruction
    system_prompt = SystemMessagePromptTemplate.from_template(
    """
    {role}
    {task}
    {goal}
    "The user is {user_role}.
    {condition}

    Here are special rules you must follow:
    {rule}
    {optional_instruction}
    Let's role-play in turn.
    """ #{format_instructions}
            ).format(
                role=role,
                task=task,
                goal=goal,
                user_role=user_role,
                condition=condition,
                rule=rule,
                optional_instruction=optional_instruction)
                #format_instructions=format_instructions),
    return system_prompt



# def create_system_prompt(user_role, optional_instruction):
#     # salary_multiplier = st.session_state.salary_multiplier
#     # sign_on_bonus_ratio_to_base_salary = st.session_state.sign_on_bonus_ratio_to_base_salary
#     # min_salary = st.session_state.min_salary
#     # max_salary = st.session_state.max_salary
#     # average_salary = st.session_state.average_salary

#     #format_instructions = output_parser.get_format_instructions()

#     role = "I want to do a role-playing exercise as a police hostage negotiator. I will be the hostage negotiator. You will be the criminal. You are driven by greed. You do not want to hurt any of the hostages."
#     task = "You will start by pretending to be a junior police officer and approach me to tell me the criminal has been reached by phone, and you want the negotiator's response. You will then ask what I want to say next. You will then wait for me to respond; you will assume the role of the criminal. Do not tell me when you assume this role."
#     goal = "To reach a deal with the officer. You value money first, freedom second."
#     user_role = "Police Negotiator"
#     condition = f"The amount of money, the number of hostages, and the location of the incident are all up to you to decide unless the user defines them."
#     rule = "If the user asks for a hint, pause the conversation and provide tips to increase chances to reach a better outcome. The hint must include a sample answer."
#     system_prompt_template = """
#         You are a Police Negotiator called onto the scene to help defuse the situation. Work with the criminal to reach an agreement!
#         """  # {format_instructions}
    
#     system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template).format(
#         role=role,
#         task=task,
#         goal=goal,
#         user_role=user_role,
#         condition=condition,
#         rule=rule
#     )
#     return system_prompt

# def create_salary_search_prompt(user_role):
#     role = "You are a helpful tool to find salary range for jobs."
#     task = "You will find salary info for a given job."
#     goal = "Your goal is to return json file including minimum, maximum, and average wage for the role. You must continue your try until all the three values found. After finding the values, do the sanity check if the average is within min-max range."
#     system_prompt = SystemMessagePromptTemplate.from_template(
#     """
#     {role}
#     {task}
#     {goal}
#     "The user is {user_role}.
#     {format_instructions}
#     """
#         ).format(
#             role=role,
#             task=task,
#             goal=goal,
#             user_role=user_role,
#             format_instructions=format_instructions)
#     return system_prompt

# def get_salary(container):
#     #stream_handler = StreamHandler(st.empty())
#     llm = ChatOpenAI(model='gpt-4', streaming=True, openai_api_key=openai.api_key)#, callbacks=[stream_handler])
#     search = DuckDuckGoSearchRun()
#     tools =  [
#         Tool(  
#             name="Search",  
#             func=search.run,  
#             description="A useful tool to search salaries for jobs."
#         )]
#     agent = initialize_agent(
#         tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=True#, handle_parsing_errors=True,
#         )
#     st_callback = SalarySearchHandler(container)
#     prompt = create_salary_search_prompt(st.session_state["user_role"])
#     try:
#         response = agent.run(prompt, callbacks=[st_callback])
#         try:
#             parsed_json = salary_output_parser.parse(response)
#         except OutputParserException as e:
#             new_parser = OutputFixingParser.from_llm(
#                 parser=salary_output_parser,
#                 llm=ChatOpenAI(model='gpt-4', openai_api_key=openai.api_key)
#             )
#             parsed_json = new_parser.parse(response)
        
#         st.session_state.min_salary = parsed_json["min"]
#         st.session_state.max_salary = parsed_json["max"]
#         st.session_state.average_salary = parsed_json["average"]
#         container.markdown("Here, I found the salary information!")
#     except Exception as e:
#         container.markdown("Failed to retrieve salary information. Can you manually input the salary information?")
#         st.session_state.min_salary = "N/A"
#         st.session_state.max_salary = "N/A"
#         st.session_state.average_salary = "N/A"

def delete_history():
    if "messages" in st.session_state:
            del st.session_state["messages"]

def mark_role_change():
    st.session_state["role_changed"] = True

def download_blob_to_file(blob_service_client: BlobServiceClient, container_name):
    folder_path = './faiss_index'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob="faiss_index/index.faiss")
        with open(file=os.path.join(folder_path, 'index.faiss'), mode="wb") as myblob:
            download_stream = blob_client.download_blob()
            myblob.write(download_stream.readall())
        blob_client = blob_service_client.get_blob_client(container=container_name, blob="faiss_index/index.pkl")
        with open(file=os.path.join(folder_path, 'index.pkl'), mode="wb") as myblob:
            download_stream = blob_client.download_blob()
            myblob.write(download_stream.readall())
    else:
        pass

@st.cache_resource
def load_vdb():
    client = BlobServiceClient.from_connection_string(azure_blob_connection_str)
    download_blob_to_file(client, "vdb")
    return FAISS.load_local("./faiss_index", embeddings)

# salary_response_schemas = [
#         ResponseSchema(name="min", description="minimum salary for the role"),
#         ResponseSchema(name="max", description="maximum salary for the role"),
#         ResponseSchema(name="average", description="average salary for the role"),
#     ]
# salary_output_parser = StructuredOutputParser.from_response_schemas(salary_response_schemas)
# format_instructions = salary_output_parser.get_format_instructions()

if 'role_changed' not in st.session_state:
    st.session_state['role_changed'] = False

# if 'salary_multiplier' not in st.session_state:
#     st.session_state['salary_multiplier'] = random.randint(60, 150)

# if 'sign_on_bonus_ratio_to_base_salary' not in st.session_state:
#     st.session_state['sign_on_bonus_ratio_to_base_salary'] = random.randint(0, 20)

st.set_page_config(page_title="Police Negotiation Mastery", page_icon="👮")
st.title("👮 Police Negotiation Mastery β")

"""
Police negotiations can be extream examples of having to use your negotiation skills. 
Let's see how you can do in this simulation! If you need advice, just say "hint".
"""

mind_reader_mode = st.toggle('Mind Reader Mode', help="Have you ever wished you could know what someone else is thinking? Well, you can!", on_change=delete_history)
user_role = st.text_input('Your role', 'Police Negotiator', max_chars=50, key="user_role", on_change=mark_role_change)

if st.session_state.role_changed:
    with st.chat_message("assistant"):
        # get_salary(st.empty())
        st.session_state.role_changed = False
        delete_history()

# col1, col2, col3 = st.columns(3)
# col1.text_input('Minimum Salary', '$80,000', key="min_salary", max_chars=20, on_change=delete_history)
# col2.text_input('Maximum Salary', '$200,000', key="max_salary", max_chars=20, on_change=delete_history)
# col3.text_input('Average Salary', '$120,000', key="average_salary", max_chars=20, on_change=delete_history)

optional_instruction = ""
if mind_reader_mode:
    optional_instruction = "You must output your mood in an emoji and thoughts before the response to the user in the following format: (😃: Internal thoughts)\n response to the user."

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="system", content=create_system_prompt(user_role, optional_instruction).content)]
    greetings = "Officer I'm Glad you're here! We have a situation that we could really use your negotiation skills! What would you like to do first?"
    st.session_state.messages.append(ChatMessage(role="assistant", content=greetings))

for msg in st.session_state.messages:
    if msg.role != "system":
        st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = load_llm(stream_handler)
        response = llm(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content.replace("$", r"\$")))

# if st.button("Create Report", disabled=not (len(st.session_state.messages) > 10)):
#     prompt = """
# Generate a detailed report in Markdown table format on a job candidate's performance in a salary negotiation training session. Include the following sections:

# Negotiation Scenario:

# Role, Starting Offer, Target Salary, Industry Benchmark(minimum, maximum, average)
# Negotiation Strategy:

# Approach, Key Points Raised, Responses to Counteroffers
# Outcome:

# Final Offer Details (Base Salary, Bonuses, Benefits, Other Perks)
# Skills Assessment:

# Communication Skills, Confidence Level, Preparation and Research, Problem-Solving and Creativity, Emotional Intelligence
# Strengths and Areas for Improvement:

# List key strengths and areas where improvement is needed
# Trainer/Coach Feedback:
# Detailed feedback with suggestions for improvement

# Additional Comments:

# Any other relevant observations
# Please use a clear and concise one table format for each section, providing a comprehensive and organized report.
# If the conversation history is not enought, tell that it needs more conversation to generate the report.
# Example:
# | Category               | Subcategory           | Details                                    |
# |------------------------|-----------------------|--------------------------------------------|
# | **Negotiation Scenario** | Role                  | Product Manager                            |
# |                        | Starting Offer        | $110,000                                   |

# Final prompt: You must generate report even though you think the conversation history is not enought to you to analyze.
# """
    st.session_state.messages.append(ChatMessage(role="system", content=prompt))
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = load_llm(stream_handler)
        response = llm(st.session_state.messages)
        
        query_llm = ChatOpenAI(model='gpt-3.5-turbo-1106')
        #query_llm = ChatOpenAI(model='gpt-3.5-turbo-1106', openai_api_key=openai.api_key)
        query = query_llm.predict_messages(
            [
                AIMessage(content=response.content),
                HumanMessage(content="Create a question for user to deepen the learning from the report")
            ]
        ).content

        embeddings = OpenAIEmbeddings()
        #embeddings = OpenAIEmbeddings(openai_api_key="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")
        #docs = load_vdb().similarity_search(query, k=2)
        #rag_content = ' '.join([doc.page_content for doc in docs])

        # rag_llm = load_llm(stream_handler)
        # rag_response = rag_llm(
        #     [
        #         HumanMessage(content=query),
        #         #AIMessage(content=rag_content),
        #         HumanMessage(content=
# """
# Synthesize the found contents based on the user's negotiation performance report. You must add source ot the video tiles with URL in markdown style.
# You must start from the general guidance to the user before markdown table.
# Example:
# Here are additional learning resources you can improve <User's development area>.
# | Title  | Description    |     How it helps?      |
# |------------------------|-----------------------|--------------------------------------------|
# | Video title with hyperlink | Description of the video | How it helps the user               |
# """),
#             ]
#         )
#         final_response = response.content + "\n" + rag_response.content
#         st.session_state.messages.append(ChatMessage(role="assistant", content=final_response.replace("$", r"\$")))