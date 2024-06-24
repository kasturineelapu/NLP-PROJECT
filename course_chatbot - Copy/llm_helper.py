from langchain_community.llms import GooglePalm
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
palm.configure(api_key="GOOGLE_API_KEY")
llm = GooglePalm()
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)



loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")

# Store the loaded data in the 'data' variable
data = loader.load()
embeddings = HuggingFaceEmbeddings()
text = "What is your refund policy?"
query_result = embeddings.embed_query(text)
embeddings = HuggingFaceEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()

vectordb = FAISS.from_documents(documents=data,
                                 embedding=embeddings)
  # Save vector database locally
vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


if __name__ == "main ":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))
    