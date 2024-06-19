import os
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from dotenv import load_dotenv

from utils import FnRetrieverOpenAIAgent

if __name__ == '__main__':

    load_dotenv()

    openai_key = os.getenv('OPENAI_API_KEY')

    def api_to_fetch_madical_data():
        response_data = [
            {
                "id": 1,
                "name": "Paracetamol",
                "description": "Paracetamol is a common pain reliever and fever reducer. It is used to treat mild to moderate pain and reduce fever.",
                "effect": "Works by increasing the pain threshold and reducing fever through its action on the hypothalamic heat-regulating center. Generally well-tolerated when used at recommended doses."
            },
            {
                "id": 2,
                "name": "Amoxicillin",
                "description": "Amoxicillin is an antibiotic used to treat bacterial infections, such as respiratory infections, ear infections, and urinary tract infections.",
                "effect": "Kills bacteria or stops their growth. It is effective against a wide range of bacterial infections but may cause side effects like diarrhea or allergic reactions."
            },
            {
                "id": 3,
                "name": "Lisinopril",
                "description": "Lisinopril is an angiotensin-converting enzyme (ACE) inhibitor used to treat high blood pressure, heart failure, and improve survival after a heart attack.",
                "effect": "Lowers blood pressure by relaxing blood vessels. Side effects can include dizziness, headache, and a dry cough."
            },
            {
                "id": 4,
                "name": "Albuterol",
                "description": "Albuterol is a bronchodilator used to relieve symptoms of asthma and chronic obstructive pulmonary disease (COPD), such as wheezing and shortness of breath.",
                "effect": "Relaxes muscles in the airways, making it easier to breathe. It works quickly, but side effects can include tremors and increased heart rate."
            },
            {
                "id": 5,
                "name": "Omeprazole",
                "description": "Omeprazole is a proton pump inhibitor (PPI) used to reduce stomach acid and treat conditions such as gastroesophageal reflux disease (GERD) and stomach ulcers.",
                "effect": "Reduces the amount of acid produced in the stomach. It can help relieve symptoms of acid reflux and ulcers, but may cause side effects like headache and nausea."
            }
        ]

        return response_data


    def api_to_fetch_finance_data():
        response_data = [
            {
                "quarter": "Q1 2023",
                "months": ["January", "February", "March"],
                "total_revenue": 1250000,
                "total_expense": 900000
            },
            {
                "quarter": "Q2 2023",
                "months": ["April", "May", "June"],
                "total_revenue": 1350000,
                "total_expense": 950000
            },
            {
                "quarter": "Q3 2023",
                "months": ["July", "August", "September"],
                "total_revenue": 1400000,
                "total_expense": 980000
            },
            {
                "quarter": "Q4 2023",
                "months": ["October", "November", "December"],
                "total_revenue": 1450000,
                "total_expense": 1000000
            }
        ]

        return response_data


    medicines_tool = FunctionTool.from_defaults(fn=api_to_fetch_madical_data)
    finance_tool = FunctionTool.from_defaults(fn=api_to_fetch_finance_data)
    llm = OpenAI(api_key=openai_key, model="gpt-3.5-turbo-1106")

    medicines_agent = OpenAIAgent.from_tools([medicines_tool], llm=llm, verbose=True,
                                             system_prompt=f"""\
                                             You are a specialized agent designed to provide medicines data. medications like Paracetamol, Amoxicillin, Lisinopril, Albuterol, Omeprazole. 
                                             You can provide details about their uses, effects, and more..
                                             You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
                                             """
                                             )

    finance_agent = OpenAIAgent.from_tools([finance_tool], llm=llm, verbose=True,
                                           system_prompt=f"""\
                                            You are a specialized agent designed to provide quarterly financial information for the year 2023.. 
                                            You can provide details about revenue and expenses for specific quarters like Q1 2023, Q2 2023, etc.
                                            You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
                                            """
                                           )

    agents = {
        "medicines_agent": medicines_agent,
        "finance_agent": finance_agent,
    }

    # Define a tool for each function agent
    query_engine_tools = []
    for agent_name, agent in agents.items():
        tool = QueryEngineTool(
            query_engine=agent,
            metadata=ToolMetadata(
                name=f"tool_{agent_name}",
                description=f"This tool uses the {agent_name} agent to answer queries.",
            ),
        )
        query_engine_tools.append(tool)

    # Define a mapping of tools to nodes
    tool_mapping = SimpleToolNodeMapping.from_objects(query_engine_tools)

    # Create an object index from the tools
    obj_index = ObjectIndex.from_objects(
        query_engine_tools,
        tool_mapping
    )
    # Instantiate a retriever over the object index
    retriever = obj_index.as_retriever(similarity_top_k=3)

    # Create the top-level agent using the retriever
    top_agent = FnRetrieverOpenAIAgent.from_retriever(
        retriever,
        system_prompt=f""" \
    You are a top-level agent designed to choose the most appropriate agent of the 5 agents provided in the object index based on the user query and use the appropriate agent to answer queries about freight and cars.
    Please ALWAYS choose the approprate agents among the {len(agents.keys())} provided based on the user query to answer a question. Do NOT rely on prior knowledge.\
    """,
        verbose=True,
    )

    # Medicines
    question1 = "What are the effects of Paracetamol?"
    question2 = "Can you list the uses and side effects of Amoxicillin?"
    question3 = "What conditions does Lisinopril treat, and what are its common side effects?"
    question4 = "Describe how Albuterol works to relieve asthma symptoms."
    question5 = "What is Omeprazole used for, and what potential side effects might it have?"
    question6 = "How does Paracetamol reduce pain and fever?"

    # Finance
    question7 = "What was the total revenue for Q2 2023, and how does it compare to Q1 2023?"
    question8 = "Provide the total expenses for each quarter in 2023."
    question9 = "How did the revenue change from Q3 2023 to Q4 2023?"
    question10 = "Which quarter had the highest total revenue and what was the amount?"
    question11 = "Calculate the total expenses for the first half of 2023."

    # Mixed Questions
    question12 = "List effects of Paracetamol along with total expenses for each quarter in 2023."

    # Query the top agent for freight rates
    response = top_agent.query(question12)
    print("Final response:")
    print(response)
