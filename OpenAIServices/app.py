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

@app.route("/", methods=['GET'])
def home():
    return "!Hello World"

@app.route("/getresponseai", methods=['GET'])
def get_response():

    os.environ['OPENAI_API_KEY'] = 'sk-iRBYXQv18QSSc9djJLzm5GkQE_6SHYxtzw3pT2YgxLT3BlbkFJwCo0L0zN5r7A7owt41wTYlXbFiWnaZj2-8qEHbGY4A'
    prompt_user = request.args.get('prompt_user')
    print(prompt_user)
    client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])

    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {"role": "system", 'content':'Tu eres un asistente que va a resolver las dudas de un usuario' },
            {"role": "user", 'content': prompt_user}

        ]
    )

    return completion.choices[0].message.content

@app.route("/getresponseairag", methods=['GET'])
def get_response_rag():
    os.environ['OPENAI_API_KEY'] = 'sk-iRBYXQv18QSSc9djJLzm5GkQE_6SHYxtzw3pT2YgxLT3BlbkFJwCo0L0zN5r7A7owt41wTYlXbFiWnaZj2-8qEHbGY4A'
    os.environ['PINECONE_API_KEY'] = '1aa94a6a-d80f-4e0f-b45a-094db7a33811'
    prompt_user = request.args.get('prompt_user')
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