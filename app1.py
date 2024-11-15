from PyPDF2 import PdfReader
import streamlit as st
import pandas as pd
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
from transformers import pipeline

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
st.write(f'Los 3 Chunks relacionados con la pregunta: "{pregunta}" son:')

docs = knowledge_base.similarity_search(pregunta, 3)
st.write(docs)

##################################################
# Escribir el contenido de docs en un archivo temporal
with open("mi_texto.txt", "w") as archivo:
    for elemento in docs:
        archivo.write(f"{elemento}\n")

# Leer el archivo y generar el botón de descarga en Streamlit
with open("mi_texto.txt", "r") as archivo:
    contenido = archivo.read()

# Crear el botón de descarga
st.download_button(
    label="Descargar archivo",
    data=contenido,
    file_name="mi_texto.txt",
    mime="text/plain"
)

##################################################
##################################################
##################################################

# Cargar el modelo de question-answering de Hugging Face
question_answerer = pipeline("question-answering")

##################################################
st.header("PARTE 5: Pregunta 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Por favor, detalla un poco más tu pregunta inicial?"}]

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

##################################################

def generate_response(prompt):
    # Generar respuesta utilizando el pipeline de question-answering
    result = question_answerer(question=prompt, context=docs)
    respuesta = f"Respuesta: '{result['answer']}', con una probabilidad de {round(result['score'] * 100)}%."
    return respuesta

##################################################

# Input del usuario para la pregunta
if prompt := st.chat_input():
    # Guardar la pregunta del usuario en el estado de la sesión
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    # Generar respuesta y guardarla en el estado de la sesión
    respuesta = generate_response(prompt)
    st.chat_message("assistant", avatar="🤖").write(respuesta)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
