from PyPDF2 import PdfReader
import streamlit as st
import pandas as pd
from io import StringIO

# Título de la página
st.title("PARTE 1: Extrae texto de PDF")

# Agregar un botón para cargar archivo, solo permitiendo archivos PDF
pdf_file = st.file_uploader("Cargar archivo PDF", type="pdf")

# Mostrar un mensaje si se carga el archivo correctamente
if pdf_file is not None:
    st.success("Archivo PDF cargado exitosamente")
    pdf_file_obj = open(pdf_file, 'rb')
    # Aquí puedes realizar operaciones adicionales con el archivo PDF
else:
    st.info("Por favor, sube un archivo PDF.")


#pdf_reader = PdfReader(pdf_file_obj)
