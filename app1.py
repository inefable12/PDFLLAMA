from PyPDF2 import PdfReader
import streamlit as st
import pandas as pd
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

# Título de la página
st.title("Analiza tu Artículo Científico con IA")
st.subheader("Autor: Jesus Alvarado Huayhuaz")

st.header("PARTE 1: Extracción de texto")
# Agregar un botón para cargar archivo, solo permitiendo archivos PDF
pdf_file_obj = st.file_uploader("Cargar archivo PDF", type="pdf")

# Mostrar un mensaje si se carga el archivo correctamente
if pdf_file_obj is not None:
    st.success("Archivo PDF cargado exitosamente")
    #pdf_file_obj = open(pdf_file, 'rb')
    pdf_reader = PdfReader(pdf_file_obj)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
# Configuración de límites de palabras para mostrar
    inicio = st.number_input("Palabras desde", min_value=0, max_value=len(text.split()), value=0, step=1)
    fin = st.number_input("Palabras hasta", min_value=0, max_value=len(text.split()), value=len(text.split()), step=1)

    # Asegurarse de que `inicio` y `fin` son enteros y extraer la subcadena
    palabras = text.split()  # Dividir el texto en palabras
    muestra = " ".join(palabras[int(inicio):int(fin)])  # Seleccionar las palabras del rango dado
    txt = st.text_area(
        "Texto extraido",
        muestra,
    )
    st.write(f"Escribiste {len(txt)} caracteres.")

else:
    st.info("Por favor, sube un archivo PDF.")

##################################################
##################################################
##################################################

st.header("PARTE 2: Chunks")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len
    )

chunks = text_splitter.split_text(text)
st.write("La cantidad total de Chunks es:", len(chunks))

# Entrada para seleccionar el número de chunk
chunk_num = st.number_input("Visualizar el Chunk número:", min_value=0, max_value=len(chunks) - 1, step=1)
# Mostrar el chunk seleccionado
st.write(chunks[int(chunk_num)])


##################################################
##################################################
##################################################

st.header("PARTE 3: Embeddings")

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

from langchain.vectorstores import FAISS
knowledge_base = FAISS.from_texts(chunks, embeddings)

st.write(embeddings)

##################################################
##################################################
##################################################

st.header("PARTE 4: Selección")

pregunta = st.text_input("Escribe tu pregunta (en inglés) para filtrar los chunks", "What repositories or databases are mentioned?")
st.write(f'Los 5 Chunks relacionados con la pregunta: "{pregunta}" son:')

docs = knowledge_base.similarity_search(pregunta, 5)
st.write(docs)

# Guardar la lista en un archivo .txt
with open("mi_texto.txt", "w") as archivo:
    for elemento in docs:
        archivo.write(f"{elemento}\n")

##################################################
##################################################
##################################################

st.header("PARTE 5: Pregunta 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Por favor, detalla un poco más tu pregunta inicial?"}]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

##################################################

def generate_response():
    response = ollama.chat(model='llama3.2', stream=True, messages=st.session_state.messages)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token

##################################################

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)
    st.session_state["full_message"] = ""
    st.chat_message("assistant", avatar="🤖").write_stream(generate_response)
    st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})
    


