import streamlit as st
from openai import AzureOpenAI
import pypdf
import pandas as pd
import io

# Configuração da página
st.set_page_config(page_title="ChatBot - Processo de Risco", layout="wide")
st.title("🤖 ChatBot - Análise de Processo de Risco")

# ------------------------------------------------------------------------------
# INICIALIZAÇÃO DO ESTADO DA SESSÃO (SESSION STATE)
# ------------------------------------------------------------------------------
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_context" not in st.session_state:
    st.session_state.rag_context = ""

# ------------------------------------------------------------------------------
# FUNÇÕES AUXILIARES PARA EXTRAÇÃO DE TEXTO (RAG)
# ------------------------------------------------------------------------------
def extract_text_from_file(uploaded_file) -> str:
    """Extrai texto dos arquivos suportados: PDF, TXT, CSV."""
    file_type = uploaded_file.name.split(".")[-1].lower()
    extracted_text = ""

    try:
        if file_type == "pdf":
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

        elif file_type == "txt":
            extracted_text = uploaded_file.read().decode("utf-8")

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            extracted_text = df.to_string()

    except Exception as e:
        st.error(f"Erro ao processar o arquivo {uploaded_file.name}: {e}")

    return extracted_text

# ------------------------------------------------------------------------------
# BARRA LATERAL (SIDEBAR): CONFIGURAÇÕES E CONEXÃO
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações da Conexão Azure OpenAI")

    api_key = st.text_input("API Key (Chave):", type="password", key="api_key_input")
    endpoint = st.text_input(
        "Endpoint:", 
        placeholder="https://seu-recurso.openai.azure.com/", 
        key="endpoint_input"
    )
    deployment_name = st.text_input(
        "Nome do Modelo / Deployment:", 
        placeholder="gpt-4o", 
        key="model_input"
    )
    api_version = st.text_input(
        "Versão da API (opcional):", 
        value="2024-02-15-preview", 
        key="api_version_input"
    )

    st.markdown("---")
    st.header("🎭 Persona & Conhecimento")

    persona_default = "Você é um especialista em análise de processos de risco corporativo. Analise as entradas com precisão, objetividade e foco na mitigação de riscos."
    system_persona = st.text_area("Persona / Instruções do Bot:", value=persona_default, height=120)

    uploaded_files = st.file_uploader(
        "Upload de arquivos de orientação (RAG)", 
        type=["pdf", "txt", "csv"], 
        accept_multiple_files=True
    )

    st.markdown("---")

    # Botão para Iniciar o Chat
    if st.button("🚀 Iniciar Chat", use_container_width=True):
        if not api_key or not endpoint or not deployment_name:
            st.error("Por favor, preencha todas as credenciais de conexão com a Azure.")
        else:
            # Processa arquivos de RAG
            context_accumulator = ""
            if uploaded_files:
                for file in uploaded_files:
                    context_accumulator += f"\n--- Conteúdo do arquivo {file.name} ---\n"
                    context_accumulator += extract_text_from_file(file) + "\n"

            st.session_state.rag_context = context_accumulator
            st.session_state.chat_started = True

            # Reseta o histórico se for o primeiro início
            if not st.session_state.messages:
                st.session_state.messages = []

            st.success("Chat configurado com sucesso!")

    # Botão de Novo Chat (mantém credenciais e reseta a conversa)
    if st.session_state.chat_started:
        if st.button("🔄 Novo Chat (Limpar Conversa)", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ------------------------------------------------------------------------------
# PROCESSAMENTO E EXIBIÇÃO DO CHAT
# ------------------------------------------------------------------------------
if not st.session_state.chat_started:
    st.info("👈 Preencha os dados da Azure OpenAI na barra lateral e clique em **Iniciar Chat** para começar.")
else:
    # Exibe histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada do Usuário
    if user_prompt := st.chat_input("Digite sua dúvida ou análise sobre o processo de risco..."):
        # Adiciona e exibe mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Montagem do Prompt do Sistema com Persona + RAG
        full_system_prompt = system_persona
        if st.session_state.rag_context:
            full_system_prompt += f"\n\nUtilize o contexto abaixo fornecido pelos documentos para responder às perguntas:\n{st.session_state.rag_context}"

        # Preparação das mensagens para envio ao modelo
        api_messages = [{"role": "system", "content": full_system_prompt}]
        for msg in st.session_state.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        # Chamada ao modelo Azure OpenAI
        with st.chat_message("assistant"):
            try:
                client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_version
                )

                response_stream = client.chat.completions.create(
                    model=deployment_name,
                    messages=api_messages,
                    stream=True
                )

                response_text = st.write_stream(response_stream)

                # Salva a resposta do assistente no histórico
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"Erro na conexão com Azure OpenAI: {e}")
