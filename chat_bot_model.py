from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import CTransformers 
from langchain.chains import RetrievalQA


DATABASE_PATH = 'VectStore/db_faiss'

custom_prompt_template = """Use the following piece of information to answer the user's question.
If you don't have the anser, please just say that you don't have the relevant information for the question.

Context:{context}
Question:{question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""
            
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    
    prompt =PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])
    return prompt


def load_llm_model():
    llm = CTransformers(
        model='llama-2-7b-chat.ggmlv3.q4_1.bin',
        model_type ='llama',
        max_new_tokens = 512,
        temperature = 0.5 
    )
    return llm


def retrieval_qa_chain(llm, prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever = db.as_retriever(search_kwargs={'k':2}),
        return_source_documents = True,
        chain_type_kwargs={'prompt':prompt}
        
    )
    return qa_chain

def qa_bot():
    embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs = {'device':'cpu'})
    db = FAISS.load_local(DATABASE_PATH,embedding)
    llm = load_llm_model()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt,db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response =qa_result({'query':query})
    return response


import chainlit as cl 

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome"
    await msg.update() 
    cl.user_session.set('chain', chain) 
    
    
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get('chain')
    callback_ = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=['FINAL','ANSWER']
    )
    callback_.answer_reached=True 
    res = await chain.acall(message.content,callbacks=[callback_])
    ans = res['result']
    scr = res['source_documents']
    await cl.Message(content=ans).send()