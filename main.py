
import os
from dotenv import load_dotenv
load_dotenv()


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

def splitDocument(doc):
    # loader = UnstructuredPDFLoader("../data/field-guide-to-data-science.pdf")
    # loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")
    loader = PyPDFLoader(doc)

    data = loader.load()
    print (f'There are {len(data)} document(s), each with {len(data[-1].page_content)} characters...')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(data) # Split again bc you're using PyPDFLoader (optional)

    return texts


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

def uploadEmbeddings(texts, retrain=False):
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YourAPIKey')

    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'YourAPIKey')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'YourAPIEnv')

    # Create embeddings, model="text-embedding-ada-002" (default)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    index_name = "chatterup-index" # the name of your pinecone index

    # Upload to vector store or query existing
    if retrain:
        docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    else:
        docsearch = Pinecone.from_existing_index(embedding=embeddings, index_name=index_name)

    return docsearch


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

def queryChatGPT(query, docsearch):
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YourAPIKey')

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY) # default is $0.02 per 1k tokens
    chain = load_qa_chain(llm, chain_type="stuff")

    docs = docsearch.similarity_search(query)

    answer = chain.run(input_documents=docs, question=query)
    return answer


def main():
    file = "./principles_abridged.pdf"
    query = "What is ray dalio's main argument and his reasoning for it?"

    texts = splitDocument(file)
    docsearch = uploadEmbeddings(texts, retrain=False) # True if pinecone deletes your stuff
    answer = queryChatGPT(query, docsearch)

    print("Question: ", query)
    print(answer)


if __name__ == "__main__":
    main()