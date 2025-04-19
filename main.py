import os
import openai
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.qa_generation.base import QAGenerationChain


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

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def crear_indice(textos):
  splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  chunks = []
  for texto in textos:
      partes = splitter.split_text(texto)
      chunks.extend(partes)
  embeddings = OpenAIEmbeddings()
  faiss_index = FAISS.from_texts(chunks, embeddings)
  return faiss_index

def configurar_qa(faiss_index):
  retriever = faiss_index.as_retriever(search_kwargs={"k": 2})  # o incluso k=1
  llm = OpenAI(temperature=0)
  qa = RetrievalQA.from_chain_type(
      llm=llm,
      retriever=retriever,
      chain_type="map_reduce",  # <-- aquí el cambio crítico
      return_source_documents=False
  )
  return qa

if __name__ == "__main__":
  textos = cargar_pdfs("pdfs")
  print(f"✅ Se cargaron {len(textos)} archivos PDF.")

  print("⏳ Indexando documentos con IA...")
  faiss_index = crear_indice(textos)

  qa = configurar_qa(faiss_index)
  print("🤖 Asistente legal listo. Escribe tu consulta:")

  while True:
      pregunta = input("📌 Tu pregunta legal: ")
      if pregunta.lower() in ["salir", "exit", "quit"]:
          break
      respuesta = qa.invoke({"query": pregunta})["result"]
      print(f"🧠 Respuesta:\n{respuesta}\n")
