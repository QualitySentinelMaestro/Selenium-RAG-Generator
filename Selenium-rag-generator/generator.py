import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (GROQ_API_KEY)
load_dotenv()

# --- 1. RAG Setup (Conceptual - requires actual documents and embedding) ---
# NOTE: This part is highly simplified. In a real RAG app, you'd load, split, and index documents.
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# Use a simple, in-memory vector store for this example
vectorstore = Chroma.from_texts(
    ["""
    Python Selenium Best Practices:
    Always use WebDriverWait for element interactions.
    Example: wait.until(EC.presence_of_element_located((By.ID, "username"))).send_keys("test")
    Use By.ID, By.NAME, or By.CSS_SELECTOR for stable locators.
    """]
, embedding=embeddings)


def generate_selenium_test(test_case_description: str) -> str:
    """
    Generates a Python Selenium test function using Groq and RAG context.
    """
    # 2. Retrieve Relevant Context
    retriever = vectorstore.as_retriever(k=1)
    retrieved_docs = retriever.invoke(test_case_description)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # 3. Define the Prompt Template
    template = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """
             You are an expert Python Selenium test automation engineer.
             Your task is to generate a complete, executable Python function 
             for a Selenium test case based on the user's description.
             
             Use the Groq API's fast inference to generate high-quality code.
             
             Adhere to the following rules and context:
             - **Do not** include the full setup/teardown boilerplate (like WebDriver setup). 
               Only provide the function definition, e.g., `def test_tc_001_login():`.
             - Use `from selenium.webdriver.common.by import By` for locators.
             - Use `WebDriverWait` with a 10-second timeout for all element actions.
             - **RAG Context:** {context}
             """
            ),
            ("user", 
             f"Generate a Python Selenium test function for this test case: {test_case_description}"
            )
        ]
    )

    # 4. Groq Integration and Generation
    groq_llm = ChatGroq(temperature=0.0,  model_name="llama-3.1-8b-instant" )
    
    chain = template | groq_llm
    
    response = chain.invoke({"context": context})
    return response.content

# --- Main Execution ---
input_file = "test_cases.txt"

print("Starting Groq RAG Selenium Test Generation...\n")

with open(input_file, 'r') as f:
    test_cases = [line.strip() for line in f if line.strip()]

for case in test_cases:
    print(f"--- Generating for: {case} ---")
    generated_code = generate_selenium_test(case)
    print(generated_code)
    print("\n" + "="*50 + "\n")