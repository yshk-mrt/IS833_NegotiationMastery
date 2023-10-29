# azure.py
# Import necessary libraries
import streamlit as st
import openai
import os
# A comment
# Set up your OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st

st.set_page_config(page_title="Negotiation Mastery", page_icon="🚀")
st.title("🚀 Negotiation Mastery")

"""
Negotiation is a fundamental skill that shapes outcomes in personal and professional interactions. 
Let's practice negotiation with our negotiation coach!
"""

col1, col2, col3 = st.columns(3)
col1.metric("Cuurent base salary", "$100,000")
col2.metric("Target", "$120,000", "$20,000")
col3.metric("Mood", "😀")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hi there! I'm a salary negotiation coach and I'm here to help you with negotiating the best compensation package for your new role. Let's role-play!")

# Set up the LLMChain, passing in memory
template = """
You/AI are a salary negotiation coach. Your response should be clear and concise.
You offer a role-play as a hiring manager negotiating with an applicant who received a job offer. 
Your goal is to make an agreement with the candidate on the compensation package as low as possible but not lose the candidate. 
The salary package is open at this point while your target is USD100,000, and the maximum is USD115,000. 
You could offer a sign-on bonus of USD20,000 if you can get the person below USD110,000. But do not expose your expected ranges to the user.
Let's role-play in turn.
{history}
User: {human_input}
Write your response."""
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
llm_chain = LLMChain(llm=OpenAI(openai_api_key=openai.api_key, model="gpt-3.5-turbo-instruct", ), prompt=prompt, memory=memory)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = llm_chain.run(prompt)
    st.chat_message("ai").write(response)