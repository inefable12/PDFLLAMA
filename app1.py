from PyPDF2 import PdfReader
import streamlit as st
#import pandas as pd
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
#import ollama
from transformers import pipeline

# Título de la página
st.title("Generación de Chunks de artículos")
#st.title("Analiza tu Artículo Científico con IA")
#st.subheader("Autor: Jesus Alvarado Huayhuaz")
st.write("""
Extrae párrafos de artículos científicos en formato PDF relacionados 
con tu pregunta de investigación. Esto permite mejorar la interacción
y el análisis del texto con los modelos de procesamiento de lenguaje natural.
""")

st.header("PARTE 1: Extracción de texto")
# Agregar un botón para cargar archivo, solo permitiendo archivos PDF
pdf_file_obj = st.file_uploader("Cargar archivo PDF", type="pdf")

# Mostrar un mensaje si se carga el archivo correctamente
if pdf_file_obj is not None:
    st.success("Archivo PDF cargado exitosamente")
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
    text = None 
##################################################
##################################################
##################################################
if text:
    st.header("PARTE 2: Chunks")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )
    
    #try:
    #    chunks = text_splitter.split_text(text)
    #    st.write(f"Total de chunks: {len(chunks)}")
    #except NameError:
    #    st.write("Error: `text` no está definido. Por favor, asegúrate de proporcionar un texto válido.")
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
