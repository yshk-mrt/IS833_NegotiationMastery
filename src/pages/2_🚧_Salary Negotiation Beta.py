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
    return llm

def create_system_prompt(user_role, optional_instruction):
    salary_multiplier = st.session_state.salary_multiplier
    sign_on_bonus_ratio_to_base_salary = st.session_state.sign_on_bonus_ratio_to_base_salary
    min_salary = st.session_state.min_salary
    max_salary = st.session_state.max_salary
    average_salary = st.session_state.average_salary

    #format_instructions = output_parser.get_format_instructions()

    role = "You are a salary negotiation coach interacting with the user in turn. Your response should be clear and concise, with care."
    task = "You offer a role-play as a hiring manager negotiating with an applicant who received a job offer."
    goal = "Your role's task is to reduce the compensation package as low as possible but not lose the candidate."
    #user_role = "product manager"
    condition = f"The basic salary info is available: the minimum salary is {min_salary}, the maximum salary is {max_salary}, the average salary is {average_salary}. The salary package is open at this point, but your target is {salary_multiplier} percent from the average. You could offer a sign-on bonus of {sign_on_bonus_ratio_to_base_salary} percent of base salary. But do not expose this to the user. You also have access to the user's resume and the option to use any information within it to support any arguments. The user's resume is found in {resume}."
    #condition = "The salary package is completely open at this point, but your target is USD100,000, and the maximum is USD120,000. You could offer a sign-on bonus of $20,000 if you can get the person below $110,000. But do not expose this to the user."
    rule = "If the user asks for hint, pause the conversation and provide tips to increase chances to receive the better compensation package. The hint must include a sample answer."
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

def create_salary_search_prompt(user_role):
    role = "You are a helpful tool to find salary range for jobs."
    task = "You will find salary info for a given job."
    goal = "Your goal is to return json file including minimum, maximum, and average wage for the role. You must continue your try until all the three values found. After finding the values, do the sanity check if the average is within min-max range."
    system_prompt = SystemMessagePromptTemplate.from_template(
    """
    {role}
    {task}
    {goal}
    "The user is {user_role}.
    {format_instructions}
    """
            ).format(
                role=role,
                task=task,
                goal=goal,
                user_role=user_role,
                format_instructions=format_instructions)
    return system_prompt

def get_salary(container):
    #stream_handler = StreamHandler(st.empty())
    llm = ChatOpenAI(model='gpt-4', streaming=True)#, callbacks=[stream_handler])
    search = DuckDuckGoSearchRun()
    tools =  [
        Tool(  
            name="Search",  
            func=search.run,  
            description="A useful tool to search salaries for jobs."
        )]
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True#, handle_parsing_errors=True,
        )
    st_callback = SalarySearchHandler(container)
    prompt = create_salary_search_prompt(st.session_state["user_role"])
    try:
        response = agent.run(prompt, callbacks=[st_callback])
        try:
            parsed_json = salary_output_parser.parse(response)
        except OutputParserException as e:
            new_parser = OutputFixingParser.from_llm(
                parser=salary_output_parser,
                llm=ChatOpenAI(model='gpt-4')
            )
            parsed_json = new_parser.parse(response)
        
        st.session_state.min_salary = parsed_json["min"]
        st.session_state.max_salary = parsed_json["max"]
        st.session_state.average_salary = parsed_json["average"]
        container.markdown("Here, I found the salary information!")
    except Exception as e:
        container.markdown("Failed to retrieve salary information. Can you manually input the salary information?")
        st.session_state.min_salary = "N/A"
        st.session_state.max_salary = "N/A"
        st.session_state.average_salary = "N/A"

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

salary_response_schemas = [
        ResponseSchema(name="min", description="minimum salary for the role"),
        ResponseSchema(name="max", description="maximum salary for the role"),
        ResponseSchema(name="average", description="average salary for the role"),
    ]
salary_output_parser = StructuredOutputParser.from_response_schemas(salary_response_schemas)
format_instructions = salary_output_parser.get_format_instructions()

if 'role_changed' not in st.session_state:
    st.session_state['role_changed'] = False

if 'salary_multiplier' not in st.session_state:
    st.session_state['salary_multiplier'] = random.randint(60, 150)

if 'sign_on_bonus_ratio_to_base_salary' not in st.session_state:
    st.session_state['sign_on_bonus_ratio_to_base_salary'] = random.randint(0, 20)

st.set_page_config(page_title="Salary Negotiation Mastery", page_icon="💰")
st.title("💰 Salary Negotiation Mastery β")

"""
Negotiation is a fundamental skill that shapes outcomes in personal and professional interactions. 
Let's practice negotiation with our negotiation coach! If you need advice, just say "hint".
"""

mind_reader_mode = st.toggle('Mind Reader Mode', help="Have you ever wished you could know what someone else is thinking? Well, you can!", on_change=delete_history)
user_role = st.text_input('Your role', 'Product Manager', max_chars=50, key="user_role", on_change=mark_role_change)

if st.session_state.role_changed:
    with st.chat_message("assistant"):
        get_salary(st.empty())
        st.session_state.role_changed = False
        delete_history()

col1, col2, col3 = st.columns(3)
col1.text_input('Minimum Salary', '$80,000', key="min_salary", max_chars=20, on_change=delete_history)
col2.text_input('Maximum Salary', '$200,000', key="max_salary", max_chars=20, on_change=delete_history)
col3.text_input('Average Salary', '$120,000', key="average_salary", max_chars=20, on_change=delete_history)

optional_instruction = ""
if mind_reader_mode:
    optional_instruction = "You must output your mood in an emoji and thoughts before the response to the user in the following format: (😃: Internal thoughts)\n response to the user."

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="system", content=create_system_prompt(user_role, optional_instruction).content)]
    greetings = "Hi there! I'm a salary negotiation coach and I'm here to help you with negotiating the best compensation package for your new role. Let's role-play!"
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

if st.button("Create Report", disabled=not (len(st.session_state.messages) > 10)):
    prompt = """
Generate a detailed report in Markdown table format on a job candidate's performance in a salary negotiation training session. Include the following sections:

Negotiation Scenario:

Role, Starting Offer, Target Salary, Industry Benchmark(minimum, maximum, average)
Negotiation Strategy:

Approach, Key Points Raised, Responses to Counteroffers
Outcome:

Final Offer Details (Base Salary, Bonuses, Benefits, Other Perks)
Skills Assessment:

Communication Skills, Confidence Level, Preparation and Research, Problem-Solving and Creativity, Emotional Intelligence
Strengths and Areas for Improvement:

List key strengths and areas where improvement is needed
Trainer/Coach Feedback:
Detailed feedback with suggestions for improvement

Additional Comments:

Any other relevant observations
Please use a clear and concise one table format for each section, providing a comprehensive and organized report.
If the conversation history is not enought, tell that it needs more conversation to generate the report.
Example:
| Category               | Subcategory           | Details                                    |
|------------------------|-----------------------|--------------------------------------------|
| **Negotiation Scenario** | Role                  | Product Manager                            |
|                        | Starting Offer        | $110,000                                   |

Final prompt: You must generate report even though you think the conversation history is not enought to you to analyze.
"""
    st.session_state.messages.append(ChatMessage(role="system", content=prompt))
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = load_llm(stream_handler)
        response = llm(st.session_state.messages)
        
        query_llm = ChatOpenAI(model='gpt-3.5-turbo-1106')
        query = query_llm.predict_messages(
            [
                AIMessage(content=response.content),
                HumanMessage(content="Create a question for user to deepen the learning from the report")
            ]
        ).content

        embeddings = OpenAIEmbeddings()
        docs = load_vdb().similarity_search(query, k=2)
        rag_content = ' '.join([doc.page_content for doc in docs])

        rag_llm = load_llm(stream_handler)
        rag_response = rag_llm(
            [
                HumanMessage(content=query),
                AIMessage(content=rag_content),
                HumanMessage(content=
"""
Synthesize the found contents based on the user's negotiation performance report. You must add source ot the video tiles with URL in markdown style.
You must start from the general guidance to the user before markdown table.
Example:
Here are additional learning resources you can improve <User's development area>.
| Title  | Description    |     How it helps?      |
|------------------------|-----------------------|--------------------------------------------|
| Video title with hyperlink | Description of the video | How it helps the user               |
"""),
            ]
        )
        final_response = response.content + "\n" + rag_response.content
        st.session_state.messages.append(ChatMessage(role="assistant", content=final_response.replace("$", r"\$")))

# PDF uploader
resume = ""

uploaded_file = st.sidebar.file_uploader("Upload your Resume (PDF)", type=['pdf'])

if uploaded_file is not None:
    pdf_file = uploaded_file.read()
    # perform operation on pdf_file
    resume += pdf_file
