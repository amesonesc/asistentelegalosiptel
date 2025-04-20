import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_PORT"] = "8000"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

import PyPDF2
import tiktoken
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

# 2. Crear índice FAISS con metadata de archivo
def crear_indice(textos, nombres_archivos):
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    documentos = []
    for i, texto in enumerate(textos):
        partes = splitter.split_text(texto)
        for parte in partes:
            documentos.append(Document(
                page_content=parte,
                metadata={"source": nombres_archivos[i]}
            ))
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documentos, embeddings)

# 3. Configurar asistente con método compatible
@st.cache_resource
def configurar_qa(_faiss_index):
    retriever = _faiss_index.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="map_reduce", verbose=False)
    return retriever, qa_chain

# 4. Limitar tokens reales
def limitar_tokens(texto, modelo="gpt-3.5-turbo", max_tokens=7000):
    enc = tiktoken.encoding_for_model(modelo)
    tokens = enc.encode(texto)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

# 5. Ejecutar consulta y mostrar múltiples fuentes
def ejecutar_consulta(retriever, qa_chain, pregunta):
    docs = retriever.invoke(pregunta)
    if not docs:
        return "⚠️ No se encontraron documentos relevantes."

    texto_total = ""
    fuentes = list(set([doc.metadata.get("source", "(sin nombre)") for doc in docs]))
    for doc in docs:
        texto_total += doc.page_content

    texto_reducido = limitar_tokens(texto_total)
    docs_reducidos = [Document(page_content=texto_reducido)]
    respuesta = qa_chain.invoke({"input_documents": docs_reducidos, "question": pregunta})
    fuente_str = "🗂️ Fuentes:\n- " + "\n- ".join(fuentes)
    return respuesta["output_text"] + f"\n\n{fuente_str}"

# ------------------------- UI Streamlit -------------------------
st.set_page_config(page_title="Asistente Legal OSIPTEL", page_icon="📄")
st.title("📄 Asistente Legal OSIPTEL")

# Subida de nuevos archivos
st.sidebar.header("📤 Subir nuevos archivos PDF")
subidos = st.sidebar.file_uploader("Agrega más resoluciones PDF", type="pdf", accept_multiple_files=True)

if subidos:
    for archivo in subidos:
        ruta = os.path.join("pdfs", archivo.name)
        with open(ruta, "wb") as f:
            f.write(archivo.read())
    st.sidebar.success(f"Se cargaron {len(subidos)} archivos nuevos. Reinicia la app o reindexa para incluirlos.")

# Reindexar manualmente
if st.sidebar.button("🔄 Reindexar ahora"):
    textos, nombres_archivos = cargar_pdfs_con_nombres("pdfs")
    faiss_index = crear_indice(textos, nombres_archivos)
    faiss_index.save_local("indice_osiptel")
    st.session_state.faiss_index = faiss_index
    st.session_state.retriever, st.session_state.qa_chain = configurar_qa(faiss_index)
    st.sidebar.success("Índice reindexado correctamente ✅")

# Cargar índice al iniciar
if "faiss_index" not in st.session_state:
    with st.spinner("Cargando y procesando documentos PDF..."):
        textos, nombres_archivos = cargar_pdfs_con_nombres("pdfs")
        faiss_index = crear_indice(textos, nombres_archivos)
        faiss_index.save_local("indice_osiptel")
        st.session_state.faiss_index = faiss_index
        st.session_state.retriever, st.session_state.qa_chain = configurar_qa(faiss_index)
else:
    try:
        st.session_state.faiss_index = FAISS.load_local("indice_osiptel", OpenAIEmbeddings())
        st.session_state.retriever, st.session_state.qa_chain = configurar_qa(st.session_state.faiss_index)
    except:
        st.error("Error al cargar el índice guardado. Intenta reiniciar o reindexar.")

st.markdown("Escribe tu consulta legal basada en resoluciones OSIPTEL 👇")

pregunta = st.text_input("🔍 Tu pregunta legal:", key="pregunta")

if pregunta:
    with st.spinner("Analizando jurisprudencia..."):
        try:
            respuesta = ejecutar_consulta(st.session_state.retriever, st.session_state.qa_chain, pregunta)
            st.success("Respuesta:")
            st.markdown(respuesta)
        except Exception as e:
            st.error(f"❌ Error: {e}")
