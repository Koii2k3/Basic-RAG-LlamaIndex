import os
import shutil
import streamlit as st
from huggingface_hub import login
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import get_response_synthesizer
from llama_index.core import Settings

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
TOP_K = 10
SIMILARITY_CUTOFF = 0.6
MAX_SELECTED_NODES = 5
TEMP_FILES_DIR = "./temp_files"

# Add Favicon
st.set_page_config(
    page_title="AIVN - RAG with Llama Index",
    page_icon="./static/aivn_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add logo
st.image("./static/aivn_logo.png", width=300)

if 'run_count' not in st.session_state:
    st.session_state['run_count'] = 0

# Tăng số lần chạy mỗi khi ứng dụng được cập nhật
st.session_state['run_count'] += 1
if st.session_state['run_count'] == 1:
    if os.path.exists(TEMP_FILES_DIR):
        shutil.rmtree(TEMP_FILES_DIR)
    os.makedirs(TEMP_FILES_DIR, exist_ok=True)
    st.cache_resource.clear()
    
# st.write(f"Ứng dụng đã chạy {st.session_state['run_count']} lần.")


class SortedRetrieverQueryEngine(RetrieverQueryEngine):
    def retrieve(self, query):
        nodes = self.retriever.retrieve(query)
        filtered_nodes = [node for node in nodes if node.score >= SIMILARITY_CUTOFF]
        sorted_nodes = sorted(filtered_nodes, key=lambda node: node.score, reverse=True)
        return sorted_nodes[:MAX_SELECTED_NODES]

st.title("Retrieval-Augmented Generation (RAG) Demo")

hf_api_key_placeholder = st.empty()
hf_api_key = hf_api_key_placeholder.text_input("Enter your Hugging Face API Key", type="password", placeholder="hf_...", key="hf_api_key")
st.markdown("Don't have an API key? Get one [here](https://huggingface.co/settings/tokens) (**Read Token** is enough)")

if hf_api_key:
    @st.cache_resource
    def load_models(hf_api_key):
        login(token=hf_api_key)
        with st.spinner("Loading models from Hugging Face..."):
            llm = HuggingFaceInferenceAPI(
                model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_api_key)
            embed_model = HuggingFaceEmbedding(model_name=f'BAAI/bge-small-en-v1.5', token=hf_api_key)
        return llm, embed_model
    
    llm, embed_model = load_models(hf_api_key)

    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, key="uploaded_files")
    if uploaded_files:
        @st.cache_resource
        def uploading_files(uploaded_files, num_documents):
            with st.spinner("Processing uploaded files..."):
                file_paths = []
                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = os.path.join(TEMP_FILES_DIR, uploaded_file.name)
                    file_paths.append(file_path)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                st.write(f"Uploaded {len(uploaded_files)}/{num_documents} files")
                
                return SimpleDirectoryReader(TEMP_FILES_DIR).load_data()
        
        num_documents = len(uploaded_files)
        documents = uploading_files(uploaded_files, num_documents)

        @st.cache_resource
        def indexing(_documents, _embed_model, num_documents):
            with st.spinner("Indexing documents..."):
                text_splitter = SentenceSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                Settings.text_splitter = text_splitter

            st.write(f"Indexing {num_documents} documents")
            return VectorStoreIndex.from_documents(
                    _documents, transformations=[text_splitter], embed_model=_embed_model, show_progress=True
                )
        
        index = indexing(documents, embed_model, num_documents)

        @st.cache_resource
        def create_retriever_and_query_engine(_index, _llm, num_documents):
            retriever = VectorIndexRetriever(
                index=_index, similarity_top_k=TOP_K)

            response_synthesizer = get_response_synthesizer(llm=_llm)

            st.write(f"Querying with {num_documents} nodes")
            
            return SortedRetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[],
            )
        
        query_engine = create_retriever_and_query_engine(index, llm, len(index.docstore.docs))
        
        query = st.text_input("Enter your query for RAG", key="query")

        if query:
            with st.spinner("Querying..."):
                response = query_engine.query(query)
                retrieved_nodes = response.source_nodes

                st.markdown("### Retrieved Documents")
                for i, node in enumerate(retrieved_nodes):
                    with st.expander(f"Document {i+1} (Score: {node.score:.4f})"):
                        st.write(node.text)

                st.markdown("### RAG Response:")
                st.write(response.response)
    
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class="footer">
        2024 AI VIETNAM | Made by <a href="https://github.com/Koii2k3/GradientVanishing" target="_blank">Koii2k3</a>
    </div>
    """,
    unsafe_allow_html=True
)