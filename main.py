import os
import openai
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

import os
import PyPDF2

def cargar_pdfs(directorio):
    textos = []
    for archivo in os.listdir(directorio):
        if archivo.endswith(".pdf"):
            ruta = os.path.join(directorio, archivo)
            with open(ruta, 'rb') as f:
                lector = PyPDF2.PdfReader(f)
                texto = ""
                for pagina in lector.pages:
                    contenido = pagina.extract_text()
                    if contenido:
                        texto += contenido
                textos.append(texto)
    return textos

if __name__ == "__main__":
  textos = cargar_pdfs("pdfs")  # Usa el nombre de la carpeta donde subiste tus archivos
  print(f"Se cargaron {len(textos)} archivos PDF.")
  print("Primeras líneas del primer archivo:\n")
  print(textos[0][:500])  # Muestra los primeros 500 caracteres del primer PDF
