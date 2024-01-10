from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai 

from dotenv import load_dotenv
import streamlit as st 
import os 

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.1,
    convert_system_message_to_human=True
)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

template1 = """
I have a problem for you to solve, the problem is {input}
Provide {number} distinct solutions and I want you to take into consideration, factors such as {factors}
"""

prompt1 = PromptTemplate(
    input_variables=["input", "factors", "number"],
    template=template1
)

chain1= LLMChain(
    llm=llm,
    prompt=prompt1,
    output_key="prop_soln"
)


template2 = """
For each of the proposed solution, evaluate their potential.
Consider their pros and cons, initial effort required, implementation, difficulty, potential callenges, and the expected outcomes.
Assign a probability of success and a confidence level to each option based on their factors
{prop_soln}
"""

prompt2 = PromptTemplate(
    input_variables=["prop_soln"],
    template=template2
)

chain2 = LLMChain(
    llm = llm,
    prompt=prompt2,
    output_key="solns"
)


template3 = """
For each solution, elaborate on the thought process by generating potential scenarios, outlining strategies for implementation,
identifying necessary partnership or resources, and proposing solutions to potential obstacles.
Additionally, consider any unexpected outcomes and outline contingency plans for their management.
{solns}
"""

prompt3 = PromptTemplate(
    input_variables=["solns"],
    template=template3
)

chain3 = LLMChain(
    llm=llm,
    prompt=prompt3,
    output_key="proc_output"
)


template4 = """
Rank the solutions based on evaluations and scenarios, assigning a probability of success in percentage for each.
Provide justification and final thought for each ranking.
Each ranking should be broken down into 4 points, Probability of success, justification, modes of failure and final thoughts.
Rank according to the highest probability of success.
{proc_output}
"""

prompt4 = PromptTemplate(
    input_variables=["proc_output"],
    template=template4
)

chain4 = LLMChain(
    llm=llm,
    prompt=prompt4,
    output_key="result"
)

chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=["input", "factors", "number"],
    output_variables=["result"]
)

st.header("Gemini ToT")

inp = st.text_input("Input", placeholder="Input", label_visibility='visible')
factors = st.text_input("Factors concerning the input", placeholder="Factors", label_visibility='visible')
num = st.slider("How many distinct solutions do you want ?", 2, 5, step=1)

if st.button("THINK", use_container_width=True):
    res = chain({"input" : inp, "factors" : factors, "number" : num})

    st.write("")
    st.write(":blue[Response]")
    st.write("")

    st.markdown(res['result'])