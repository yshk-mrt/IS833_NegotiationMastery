import streamlit as st
import openai
import os
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import streamlit as st

openai.api_key = os.environ.get('OPENAI_API_KEY')

response_schemas = [
    ResponseSchema(name="thought", description="internal thoughts to the user's question"),
    ResponseSchema(name="mood", description="an emoji to express your mood"),
    ResponseSchema(name="answer", description="answer to the user's question"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

def load_llm():
    llm = ChatOpenAI(model='gpt-4')
    return llm

role = "You are a salary negotiation coach interacting with the user in turn. Your response should be clear and concise, with care."
task = "You offer a role-play as a hiring manager negotiating with an applicant who received a job offer."
goal = "Your role's task is to reduce the compensation package as low as possible but not lose the candidate."
user_role = "The user is product manager."
condition = "The salary package is completely open at this point, but your target is $100,000, and the maximum is $120,000. You could offer a sign-on bonus of $20,000 if you can get the person below $110,000. But do not expose this to the user."
rule = "If the user asks for tips, pause the conversation and give him a tip. The tip should include a sample answer."
optional_instruction = ""

llm = load_llm()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
"""
{role}
{task}
{goal}
{user_role}
{condition}

Here are special rules you must follow:
{rule}
{optional_instruction}
{format_instructions}
Let's role-play in turn.
"""
        ).format(
            role=role,
            task=task,
            goal=goal,
            user_role=user_role,
            condition=condition,
            rule=rule,
            optional_instruction=optional_instruction,
            format_instructions=format_instructions),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
st.set_page_config(page_title="Salary Negotiation Mastery", page_icon="💰")
st.title("💰 Salary Negotiation Mastery (Under Construction)")

"""
Negotiation is a fundamental skill that shapes outcomes in personal and professional interactions. 
Let's practice negotiation with our negotiation coach!
"""

col1, col2 = st.columns(2)
col1.metric("Cuurent base salary", "$100,000")
col2.metric("Target", "$120,000", "$20,000")

# Set up memory
#msgs = StreamlitChatMessageHistory(key="langchain_messages")
#memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hi there! I'm a salary negotiation coach and I'm here to help you with negotiating the best compensation package for your new role. Let's role-play!")

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = chain.run(prompt)
    parsed_json = output_parser.parse(response)
    st.chat_message("ai").write("(" + parsed_json["mood"] + ": " + parsed_json["thought"] + ") " + parsed_json["answer"])