from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

txt_loader = TextLoader("FAQ.txt")
document = txt_loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
split_document = splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  
    model_kwargs={"device": "cpu"},  
)
database = FAISS.from_documents(split_document, embeddings)

retreiver = database.as_retriever(search_kwargs={"k": 2})

class AgentState(TypedDict):
    context: list[Document]
    question: str
    answer: str

def retrieve_node(state: AgentState) -> Command:
    """Retrieves relevant data (context) from the FAQ document"""
    context = retreiver.invoke(state['question'])
    return Command(
        goto="generate",
        update={"context": context}
    )

template = """Answer the queries of user based on given context:
    {context}

    Question: {question}
    You are a helper bot"""

prompt = ChatPromptTemplate.from_template(template)

def generate_node(state: AgentState) -> Command:
    """Generates answer using the retrieved context"""
    question = state["question"]
    context = state["context"]
    context_text = "\n".join(doc.page_content for doc in context)

    answer = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "context": context_text,
        "question": question
    })

    return Command(
        goto=END,
        update={"answer": answer}
    )

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
graph = workflow.compile()

print("Chatbot -- (type 'quit' to exit)")
while True:
    question = input("\033[93;1mYou: \033[0m")
    if question.lower() == "quit":
        break
    result = graph.invoke({
        "question": question, 
        "context": [], 
        "answer": ""
    })
    print(f"\n\033[92;1mBot:\033[0m {result['answer']}")
