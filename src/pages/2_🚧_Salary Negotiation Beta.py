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
import streamlit as st

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
    format_instructions = output_parser.get_format_instructions()

    role = "You are a salary negotiation coach interacting with the user in turn. Your response should be clear and concise, with care."
    task = "You offer a role-play as a hiring manager negotiating with an applicant who received a job offer."
    goal = "Your role's task is to reduce the compensation package as low as possible but not lose the candidate."
    #user_role = "product manager"
    condition = "The salary package is completely open at this point, but your target is USD100,000, and the maximum is USD120,000. You could offer a sign-on bonus of $20,000 if you can get the person below $110,000. But do not expose this to the user."
    rule = "If the user asks for hint, pause the conversation and give them a hint. The hint should include a sample answer."
    #optional_instruction = ""
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

def clear_session():
   st.session_state.clear()

st.set_page_config(page_title="Salary Negotiation Mastery", page_icon="💰")
st.title("💰 Salary Negotiation Mastery β")

"""
Negotiation is a fundamental skill that shapes outcomes in personal and professional interactions. 
Let's practice negotiation with our negotiation coach! If you need advice, just say "hint".
"""

col1, col2 = st.columns(2)

col1.metric("Cuurent base salary", "$100,000")
col2.metric("Target", "$120,000", "$20,000")

user_role = st.text_input('Your role', 'Product Manager', max_chars=50, on_change=clear_session)
mind_reader_mode = st.toggle('Mind Reader Mode', help="Have you ever wished you could know what someone else is thinking? Well, you can!", on_change=clear_session)

optional_instruction = ""
if mind_reader_mode:
    optional_instruction = "You must output your mood in an emoji and thoughts before the response to the user in the following format: ([emoji]: [internal_thoughts])\n [response]."

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
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
    
    #st.markdown(response)
    #parsed_json = output_parser.parse(response)
    #st.chat_message("ai").write("(" + parsed_json["mood"] + ": " + parsed_json["thought"] + ") " + parsed_json["answer"]) 
    

