from PyPDF2 import PdfReader
import streamlit as st
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import string

# Título de la página
st.title("Extrae párrafos de artículos científicos")
st.subheader("Autor: Jesus Alvarado Huayhuaz")
st.write("""
Extrae párrafos en formato TXT, a partir de artículos científicos en formato PDF
relacionados con tu pregunta de investigación, para luego ser analizados con IA. 
Esto permite mejorar la interacción y el análisis del texto con los modelos PLN.
""")

st.header("1. Extracción de texto")
# Agregar un botón para cargar archivo, solo permitiendo archivos PDF
pdf_file_obj = st.file_uploader("Cargar archivo PDF", type="pdf")

# Mostrar un mensaje si se carga el archivo correctamente
if pdf_file_obj is not None:
    st.success("Archivo PDF cargado exitosamente")
    pdf_reader = PdfReader(pdf_file_obj)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    tokens = [t for t in text.split()]
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation.replace('.', ''))
    stripped = [w.translate(table) for w in tokens]
    text = ' '.join(stripped)   

    st.write("Explora el contenido por número de palabras")
# Configuración de límites de palabras para mostrar
    inicio = st.number_input("Palabras desde", min_value=0, max_value=len(text.split()), value=0, step=1)
    fin = st.number_input("Palabras hasta", min_value=0, max_value=len(text.split()), value=len(text.split()), step=1)

    # Asegurarse de que `inicio` y `fin` son enteros y extraer la subcadena
    palabras = text.split()  # Dividir el texto en palabras
    muestra = " ".join(palabras[int(inicio):int(fin)])  # Seleccionar las palabras del rango dado
    txt = st.text_area(
        "Texto contenido",
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
    st.header("2. Chunks")

    chunk_size = st.number_input("Selecciona el tamaño del chunk", min_value=100, value=1000, step=100)
    chunk_overlap = st.number_input("Selecciona el solapamiento del chunk", min_value=0, value=100, step=10)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
    
    st.header("3. Embeddings")
    
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    from langchain.vectorstores import FAISS
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    st.write(embeddings)

##################################################
##################################################
##################################################

    st.header("4. Selección")

    num_chunks = st.number_input("Selecciona la cantidad de chunks a mostrar", min_value=1, value=3, step=1)
    
    pregunta = st.text_input("Escribe tu pregunta (en inglés) para filtrar los chunks", "What repositories or databases are mentioned?")
    
    st.write(f'Los {num_chunks} Chunks relacionados con la pregunta: "{pregunta}" son:')
   
    docs = knowledge_base.similarity_search(pregunta, num_chunks)
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
