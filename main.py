import os
import PyPDF2
import tiktoken
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# 1. Leer PDFs y nombres de archivo
def cargar_pdfs_con_nombres(directorio):
    textos = []
    nombres = []
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
                nombres.append(archivo)
    return textos, nombres

# 2. Crear índice con metadatos de archivo
def crear_indice(textos, nombres_archivos):
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    documentos = []
    for i, texto in enumerate(textos):
        partes = splitter.split_text(texto)
        for j, parte in enumerate(partes):
            documentos.append(Document(
                page_content=parte,
                metadata={"source": nombres_archivos[i]}
            ))
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documentos, embeddings)

# 3. Configurar la cadena de preguntas con ChatOpenAI
def configurar_qa(faiss_index):
    retriever = faiss_index.as_retriever(search_kwargs={"k": 1})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="map_reduce", verbose=False)
    return retriever, qa_chain

# 4. Limitar tokens por seguridad
def limitar_tokens(texto, modelo="gpt-3.5-turbo", max_tokens=7000):
    enc = tiktoken.encoding_for_model(modelo)
    tokens = enc.encode(texto)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

# 5. Ejecutar consulta y devolver fuente
def ejecutar_consulta(retriever, qa_chain, pregunta):
    docs = retriever.invoke(pregunta)
    if not docs:
        return "⚠️ No se encontraron documentos relevantes."

    texto_total = ""
    fuente = docs[0].metadata.get("source", "(sin nombre)")
    for doc in docs:
        texto_total += doc.page_content

    texto_reducido = limitar_tokens(texto_total)
    docs_reducidos = [Document(page_content=texto_reducido)]
    respuesta = qa_chain.invoke({"input_documents": docs_reducidos, "question": pregunta})
    return respuesta["output_text"] + f"\n🗂️ Fuente: {fuente}"

# 6. Programa principal
if __name__ == "__main__":
    print("📂 Cargando archivos PDF...")
    textos, nombres_archivos = cargar_pdfs_con_nombres("pdfs")
    print(f"✅ Se cargaron {len(textos)} archivos.")

    print("📦 Indexando documentos...")
    faiss_index = crear_indice(textos, nombres_archivos)

    print("🧠 Configurando asistente legal...")
    retriever, qa_chain = configurar_qa(faiss_index)

    print("🤖 Asistente legal listo. Escribe tu consulta:\n")
    while True:
        pregunta = input("📌 Tu pregunta legal: ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break
        try:
            respuesta = ejecutar_consulta(retriever, qa_chain, pregunta)
            print(f"\n🧾 Respuesta:\n{respuesta}\n")
        except Exception as e:
            print(f"⚠️ Error al procesar la pregunta:\n{e}\n")
