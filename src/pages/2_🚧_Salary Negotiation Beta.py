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
#set_debug(True)

openai.api_key = os.environ.get('OPENAI_API_KEY')

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "|")
    
    def on_llm_end(self, token: str, **kwargs) -> None:
        self.container.markdown(self.text)

from typing import Any, Dict, List, Union
from langchain.schema import AgentAction
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

response_schemas = [
        ResponseSchema(name="thought", description="internal thoughts to the user's question"),
        ResponseSchema(name="mood", description="an emoji to express your mood"),
        ResponseSchema(name="answer", description="answer to the user's question"),
    ]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

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
    condition = f"The basic salary info is available: the minimum salary is {min_salary}, the maximum salary is {max_salary}, the average salary is {average_salary}. The salary package is open at this point, but your target is {salary_multiplier} percent from the average. You could offer a sign-on bonus of {sign_on_bonus_ratio_to_base_salary} percent of base salary. But do not expose this to the user."
    #condition = "The salary package is completely open at this point, but your target is USD100,000, and the maximum is USD120,000. You could offer a sign-on bonus of $20,000 if you can get the person below $110,000. But do not expose this to the user."
    rule = "If the user asks for hint, pause the conversation and give them a hint. The hint should include a sample answer."
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

salary_response_schemas = [
        ResponseSchema(name="min", description="minimum salary for the role"),
        ResponseSchema(name="max", description="maximum salary for the role"),
        ResponseSchema(name="average", description="average salary for the role"),
    ]
salary_output_parser = StructuredOutputParser.from_response_schemas(salary_response_schemas)
format_instructions = salary_output_parser.get_format_instructions()

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

def clear_session():
   st.session_state.clear()

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
        #st.session_state.min_salary = st.session_state.col_min
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

st.set_page_config(page_title="Salary Negotiation Mastery", page_icon="💰")
st.title("💰 Salary Negotiation Mastery β")

"""
Negotiation is a fundamental skill that shapes outcomes in personal and professional interactions. 
Let's practice negotiation with our negotiation coach! If you need advice, just say "hint".
"""

mind_reader_mode = st.toggle('Mind Reader Mode', help="Have you ever wished you could know what someone else is thinking? Well, you can!", on_change=delete_history)

if 'role_changed' not in st.session_state:
    st.session_state['role_changed'] = False

if 'salary_multiplier' not in st.session_state:
    st.session_state['salary_multiplier'] = random.randint(-20, 40)

if 'sign_on_bonus_ratio_to_base_salary' not in st.session_state:
    st.session_state['sign_on_bonus_ratio_to_base_salary'] = random.randint(0, 20)

def mark_role_change():
    st.session_state["role_changed"] = True

user_role = st.text_input('Your role', 'Product Manager', max_chars=50, key="user_role", on_change=mark_role_change)

if st.session_state.role_changed:
    with st.chat_message("assistant"):
        get_salary(st.empty())
        st.session_state.role_changed = False
        delete_history()
        #st.session_state.messages.append(ChatMessage(role="assistant", content=response))
        
col1, col2, col3 = st.columns(3)
col1.text_input('Minimum Salary', '$80,000', key="min_salary", max_chars=12, on_change=delete_history)
col2.text_input('Maximum Salary', '$200,000', key="max_salary", max_chars=12, on_change=delete_history)
col3.text_input('Average Salary', '$12,000', key="average_salary", max_chars=12, on_change=delete_history)

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
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content.replace("$", r"\\$")))

# PDF uploader
uploaded_file = st.sidebar.file_uploader("Upload your Resume (PDF)", type=['pdf'])

if uploaded_file is not None:
    pdf_file = uploaded_file.read()
    # perform operation on pdf_file