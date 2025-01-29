import gradio as gr 
import os

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# importing necessary functions from dotenv library
from dotenv import load_dotenv

# loading variables from .env file
load_dotenv() 

# Retrieve the environment variable to verify
hugging_face_api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]
pinecone_api_key = os.environ["PINECONE_API_TOKEN"]

pc = Pinecone(api_key=pinecone_api_key)
index_name = "chatbot"  

hf = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    temperature=0.5,
)

embeddings = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device' : 'cpu'},
    encode_kwargs = {'normalize_embeddings' : True}
)


if index_name in pc.list_indexes().names():
    # Connect to the index
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

else:
    loader = PyPDFDirectoryLoader("./Biomedical Research/")
    data = loader.load()
    text_splitter = SemanticChunker(embeddings)
    texts = text_splitter.split_documents(data)

    pc.create_index(name=index_name,
                    dimension=384,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    ))
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    #add documents id to pinecone
    uuids = [str(uuid4()) for _ in range(len(texts))]
    vector_store.add_documents(documents=texts, ids=uuids)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)


prompt = """Answer any use questions based solely on the context below:

            <context>
            {context}
            </context>"""

#Get the prompt from the Langchain hub
retrieval_qa_chat_prompt = ChatPromptTemplate([
    ("system",prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ])  

combine_docs_chain = create_stuff_documents_chain(
    
    llm=hf,
    prompt= retrieval_qa_chat_prompt, 
    output_parser=StrOutputParser(),
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)



def chat_with_model(history, new_message):
    messages = [{"role" : "system" , "content" : "You are a helpful assistant."}]
    for user_message, assistant_response in history:
        messages.append({"role" : "user" , "content" : user_message})
        messages.append({"role" : "assistant" , "content" : assistant_response})
    
    assistant_message = None
    messages.append({"role" : "user" , "content" : new_message})

    for entry in history:
        role, messages = entry
        if role == 'assistant':
            assistant_message = [{"role" : "assistant" , "content" : messages}]

    if assistant_message is not None:
        response = retrieval_chain.invoke(input={
                                        "input": new_message,
                                        "chat_history": history[{"user": new_message},{"assistant":assistant_message}],  # Add any other necessary keys here
                                        "agent_scratchpad": []  # Ensure all placeholders are provided
                                    })

        assistant_message = response['answer']

        history.append((new_message,assistant_message))

        return history , ""
    else:
        response = retrieval_chain.invoke(input={
                                    "input": new_message,
                                    "chat_history": [],  # there is no history
                                    "agent_scratchpad": []  
                                })

        assistant_message = response['answer']

        history.append((new_message,assistant_message))

        return history , ""
        

def gradio_chat_app():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# Biomedical Chatbot")
        gr.Markdown("Chat with the HuggingFace model in a conversational format.")
        chatbot = gr.Chatbot(label = "Chat Interface")
        user_input = gr.Textbox(label = "your message" , placeholder = "Type something ..." , lines=1)  
        send_button = gr.Button("send")
        
        def clear_chat():
            return [] , ""
    
        clear_button = gr.Button("Clear chat")
        
        send_button.click(
            fn=chat_with_model, 
            inputs = [chatbot, user_input],
            outputs = [chatbot, user_input]
        )
        clear_button.click(
            fn=clear_chat,
            inputs = [],
            outputs = [chatbot, user_input]
        )
        
    return app


if __name__ == "__main__":
    app = gradio_chat_app()
    app.launch()