import sys
sys.path.append('..')
from src.rag_chain import load_llm

# Just test LLM loads correctly
llm = load_llm()
response = llm.invoke("What is 2 + 2?")
print(f"LLM Response: {response}")
