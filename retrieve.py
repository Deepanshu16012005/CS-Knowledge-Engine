import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


from rag import get_rag_answer

formulator_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0, 
    max_retries=2
)

# 2. Define the Prompt Template for the reformulator
with open("./prompts/query_formulator.txt" , "r") as file:
    system_instruction = file.read()

reformulator_prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("user", "{question}")
])
reformulator_chain = reformulator_prompt | formulator_llm

chat_history=[]

print("Type 'quit' to quit, or 'clear' to wipe chat history for this session.")
print("-" * 40)
# 4. Create an interactive loop to ask questions
while True:
    query = input("\nWhat would you like to search for? \n")
    
    if query.lower() == 'quit':
        break
    if query.lower() == 'clear':
        chat_history.clear()
        print("🧹 Chat history cleared for this session.")
        continue

    history_text = ""
    if len(chat_history) > 0:
        for h in chat_history[-2:]: # Just grab the last 2 interactions
            history_text += f"User: {h['user']}\nAI: {h['ai']}\n"

    response = reformulator_chain.invoke({
        "history": history_text,
        "question": query
    })
    actual_query = response.content.strip()
    print(f"Formulated Query: {actual_query}")
    final_answer = get_rag_answer(actual_query)

    print(f"\nAnswer By AI:\n{final_answer}")

    chat_history.append({
        "user": actual_query, 
        "ai": final_answer
    })
