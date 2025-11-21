import streamlit as st
from io import BytesIO
import string
# Se recomienda usar pypdf en lugar de PyPDF2 (que está obsoleto)
from pypdf import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Importaciones actualizadas para langchain community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Título de la página
st.title("Extrae párrafos de artículos científicos")
st.subheader("Autor: Jesus Alvarado Huayhuaz")
st.write("""
Extrae párrafos en formato TXT, a partir de artículos científicos en formato PDF
relacionados con tu pregunta de investigación, para luego ser analizados con IA. 
Esto permite mejorar la interacción y el análisis del texto con los modelos PLN.
""")

# --- FUNCIÓN PARA CACHÉ DE RECURSOS ---
# Esto evita que el modelo se recargue en cada interacción, ahorrando memoria y tiempo
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

st.header("1. Extracción de texto")
# Agregar un botón para cargar archivo, solo permitiendo archivos PDF
pdf_file_obj = st.file_uploader("Cargar archivo PDF", type="pdf")

text = None

# Mostrar un mensaje si se carga el archivo correctamente
if pdf_file_obj is not None:
    st.success("Archivo PDF cargado exitosamente")
    
    try:
        pdf_reader = PdfReader(pdf_file_obj)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text

        # Limpieza de texto
        tokens = text.split()
        tokens = [w.lower() for w in tokens]
        # Eliminamos puntuación excepto puntos si es necesario, o toda como en tu código original
        table = str.maketrans('', '', string.punctuation.replace('.', ''))
        stripped = [w.translate(table) for w in tokens]
        text = ' '.join(stripped)   

        st.write("Explora el contenido por número de palabras")
        
        # Validación de límites para evitar errores si el texto es corto
        max_words = len(text.split())
        if max_words > 0:
            # Configuración de límites de palabras para mostrar
            col1, col2 = st.columns(2)
            with col1:
                inicio = st.number_input("Palabras desde", min_value=0, max_value=max_words, value=0, step=1)
            with col2:
                fin = st.number_input("Palabras hasta", min_value=0, max_value=max_words, value=max_words, step=1)

            # Asegurarse de que `inicio` y `fin` son enteros y extraer la subcadena
            palabras = text.split()  # Dividir el texto en palabras
            # Controlar que inicio sea menor que fin
            if inicio < fin:
                muestra = " ".join(palabras[int(inicio):int(fin)])  # Seleccionar las palabras del rango dado
                txt = st.text_area(
                    "Texto contenido",
                    muestra,
                    height=200
                )
                st.write(f"Visualizando caracteres: {len(txt)}")
            else:
                st.warning("El valor 'hasta' debe ser mayor que 'desde'.")
        else:
            st.warning("No se pudo extraer texto del PDF.")
            text = None

    except Exception as e:
        st.error(f"Error al leer el PDF: {e}")
        text = None

else:
    st.info("Por favor, sube un archivo PDF.")
    text = None 

##################################################
if text:
    st.header("2. Chunks")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Tamaño del chunk", min_value=100, value=1000, step=100)
    with col2:
        chunk_overlap = st.number_input("Solapamiento (Overlap)", min_value=0, value=100, step=10)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
        )
  
    chunks = text_splitter.split_text(text)
    st.write(f"La cantidad total de Chunks es: **{len(chunks)}**")
    
    if len(chunks) > 0:
        # Entrada para seleccionar el número de chunk
        chunk_num = st.number_input("Visualizar el Chunk número:", min_value=0, max_value=len(chunks) - 1, step=1)
        # Mostrar el chunk seleccionado
        st.text_area("Contenido del Chunk", chunks[int(chunk_num)], height=150)

    ##################################################
    
    st.header("3. Embeddings")
    
    with st.spinner("Cargando modelo de embeddings... esto puede tardar un poco la primera vez."):
        embeddings = get_embeddings_model()
    
    # Crear la base de conocimiento
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    st.success("Modelo de Embeddings cargado y base de datos vectorial creada.")

    ##################################################

    st.header("4. Selección")

    num_chunks = st.number_input("Selecciona la cantidad de chunks a mostrar", min_value=1, value=3, step=1)
    
    pregunta = st.text_input("Escribe tu pregunta para filtrar los chunks", "What repositories or databases are mentioned?")
    
    if pregunta:
        st.write(f'Buscando respuestas para: "{pregunta}"...')
    
        docs = knowledge_base.similarity_search(pregunta, k=num_chunks)
        
        # Mostrar resultados en pantalla
        for i, doc in enumerate(docs):
            with st.expander(f"Resultado {i+1}"):
                st.write(doc.page_content)

        ##################################################
        # Generar archivo para descargar en memoria (sin guardar en disco)
        result_text = ""
        for elemento in docs:
            result_text += f"{elemento.page_content}\n\n---\n\n"
        
        # Crear el buffer en memoria
        buffer = BytesIO()
        buffer.write(result_text.encode('utf-8'))
        buffer.seek(0)
        
        # Crear el botón de descarga
        st.download_button(
            label="Descargar resultados (.txt)",
            data=buffer,
            file_name="resultados_investigacion.txt",
            mime="text/plain"
        )
