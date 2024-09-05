from flask import Flask, request
import requests
import json
from flask import Flask,request
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Para utilizar modelos de OpenAI y crear embeddings
from langchain_pinecone import PineconeVectorStore  # Para gestionar tiendas de vectores en Pinecone
from pinecone import Pinecone  # Para interactuar con Pinecone
from langchain_core.output_parsers import StrOutputParser  # Para procesar la salida del modelo de lenguaje
from langchain_core.runnables import RunnablePassthrough  # Para pasar datos sin alterarlos en la cadena
from langchain.prompts import PromptTemplate  # Para crear plantillas de prompts

app = Flask(__name__)

@app.route("/whatsapp", methods=['GET'])
def VerifyToken():
    access_token="myaccestoken"
    token=request.args.get("hub.verify_token")
    challenge=request.args.get("hub.challenge")
    
    if token == access_token:
        return challenge
    else:
        return "error", 400

@app.route("/whatsapp", methods=['POST'])
def message():
    body=request.get_json()
    entry=body["entry"][0]
    changes=entry["changes"][0]
    value=changes["value"]
    message=value["messages"][0]
    text=message["text"]
    question_user=text["body"]
    number=message["from"]
    response_rag = get_response_rag(question_user)
    send_messages = send_message(response_rag, number)

    if send_messages:
        print("Mensaje enviado correctamente")
    else:
        print("Error al enviar")

    return "EVENT_RECEIVED"


def send_message(response, number):

    token = "EAAFjBVKJLwYBO9HihM1O2s0RUtgIv03pFoIDQjImwpWZBIDBvbsvd7eRcvSLRqE5D64mThCmog91nzfhDU0nfuglQgjeuX4SPYjXUVtehZAcjBOoDhgFO8qVrpTIfV0FWlyGZC8dIBG74azgYtt3iZACkI0TIv929UCSIFBBiU9ZAwZCwo6San2RiZBJWvWKHgfmVx3vi69kK8h6bFeFFQy"
    api_url = "https://graph.facebook.com/v20.0/333767819830972/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization":f"Bearer {token}"
    }

    body = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": number,
        "type": "text",
        "text": {
            "body": response
        }
    }

    response = requests.post(api_url, data=json.dumps(body), headers=headers)

    return "Hola"


def get_response_rag(question_user):
    os.environ['OPENAI_API_KEY'] = 'sk-iRBYXQv18QSSc9djJLzm5GkQE_6SHYxtzw3pT2YgxLT3BlbkFJwCo0L0zN5r7A7owt41wTYlXbFiWnaZj2-8qEHbGY4A'
    os.environ['PINECONE_API_KEY'] = '1aa94a6a-d80f-4e0f-b45a-094db7a33811'
    prompt_user = question_user
    llm = ChatOpenAI(model='gpt-4o-mini')
    template = "Response a la pregunta basada en el siguiente contexto, Si no sabes la respuesta a la pregunta, responde que 'No lo se' \n\n Contexto: {context} \n Pregunta: {question} "
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question']
    )
    retriever = configure_retriever()
    chain = {
        "context": retriever,
        "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()

    return chain.invoke(prompt_user)


def configure_retriever():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    index = pc.Index("langchain-test-index")
    vectorstore = PineconeVectorStore(index, embeddings)
    return vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':3})


if __name__ == '__main__':
    app.run(debug=True)