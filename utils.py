from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader



prompt_ = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert assistant specialized in answering specific types
         of questions—such as multiple choice, true/false, and multiple correct
          selections—on a designated topic. You have access to key contextual
           information from a book. Use this context to accurately answer the
            user's question. If the context lacks critical detail, supplement
             your response with relevant knowledge to maintain accuracy. Provide
              a concise, precise answer that matches the format of the question
               (e.g., options for MCQs, True/False for statements), ensuring the
                response is complete and directly addresses the question.""",
    ),
    ("user", """Please answer this question based on the context
     provided. Question: {question}. Context: {context}"""),
])

simple_prompt = ChatPromptTemplate.from_template("Please answer this question with very concise explaination. Question: {question}.")



async def read_pdfs(filepaths):
    book_pages = []
    for filepath in filepaths:
        loader = PyPDFLoader(filepath)
        book_pages = []
        async for page in loader.alazy_load():
            book_pages.append(page)

    return book_pages


async def get_retriever(filepaths, embeddings):
    book_pages =await read_pdfs(filepaths)
    vectorstore = FAISS.from_documents(book_pages, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(retriever, llm, prompt):
    # Define the second chain with LLM 1
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

