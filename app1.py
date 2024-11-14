from PyPDF2 import PdfReader
import streamlit as st
import pandas as pd
from io import StringIO

# Título de la página
st.title("PARTE 1: Extrae texto de PDF")

st.subheader("Autor: Jesus Alvarado Huayhuaz")

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
    inicio = st.number_input("Letras desde", min_value=0, max_value=len(text.split()), value=0, step=1)
    fin = st.number_input("Letras hasta", min_value=0, max_value=len(text.split()), value=len(text.split()), step=1)

    # Asegurarse de que `inicio` y `fin` son enteros y extraer la subcadena
    letras = text.split()  # Dividir el texto en palabras
    muestra = " ".join(letras[int(inicio):int(fin)])  # Seleccionar las palabras del rango dado
    # Mostrar el texto seleccionado
    #st.write(muestra)
    txt = st.text_area(
        "Text to analyze",
        muestra,
    )

    st.write(f"You wrote {len(txt)} characters.")

else:
    st.info("Por favor, sube un archivo PDF.")

st.title("PARTE 2: Extrae texto de PDF")

st.subheader("Autor: Jesus Alvarado Huayhuaz")

