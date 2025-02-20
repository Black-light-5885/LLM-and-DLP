from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

INPUT_DATA_FOLDER = 'data/'
FAISS_DATABASE_PATH = 'VectStore/db_faiss'


def create_db_faiss():
    loader = DirectoryLoader(INPUT_DATA_FOLDER,glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap = True)
    texts = text_splitter.split_documents(documents=documents)
    
    embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs = {'device':'cpu'})
    
    db= FAISS.from_documents(texts,embedding)
    db.save_local(FAISS_DATABASE_PATH)
    
if __name__ == '__main__':
    create_db_faiss()
    