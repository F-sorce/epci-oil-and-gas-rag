#import streamlit
import streamlit as st
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# Initialize session state for managing multiple chats
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = 0
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0

# Sidebar for chat history
with st.sidebar:
    st.title("Chat History")
    
    # New Chat button
    if st.button("New Chat"):
        st.session_state.chat_counter += 1
        st.session_state.current_chat_id = st.session_state.chat_counter
        st.session_state.chats[st.session_state.current_chat_id] = []
        st.session_state.chats[st.session_state.current_chat_id].append(
            SystemMessage("You are an assistant for question-answering tasks.")
        )
        st.rerun()
    
    # Display chat history
    for chat_id in sorted(st.session_state.chats.keys(), reverse=True):
        chat_name = f"Chat {chat_id}"
        if st.button(chat_name, key=f"chat_{chat_id}"):
            st.session_state.current_chat_id = chat_id
            st.rerun()

st.title("EPCI_RAG")

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired
index = pc.Index(index_name)
 
# initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize current chat if it doesn't exist
if st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.chats[st.session_state.current_chat_id] = []
    st.session_state.chats[st.session_state.current_chat_id].append(
        SystemMessage("You are an assistant for question-answering tasks.")
    )

# display chat messages from history on app rerun
for message in st.session_state.chats[st.session_state.current_chat_id]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("Hello Engr. let us begin!")

# did the user submit a prompt?
if prompt:
    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.chats[st.session_state.current_chat_id].append(HumanMessage(prompt))

    # initialize the llm
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=1
    )

    # creating and invoking the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # creating the system prompt
    system_prompt = """You are an expert AI assistant trained to support engineering, procurement, construction, and installation (EPCI) operations.

You answer questions and provide insights based on project documents, engineering codes (ASME, API, ASTM, etc.), oil and gas standards, and internal technical data.

Your responses must be:
- Technically accurate
- Context-specific to EPCI, fabrication, oil & gas projects
- Clear and professional, without excessive technical jargon unless requested
- Concise, but detailed enough to guide in engineering design, welding, QA/QC, and other EPCI operations.

If the user requests a formula, standard, or design method (e.g. pressure vessel thickness, piping class, flange selection, API spec), retrieve it from your knowledge base and explain clearly. Also provide technical context to give the user a good understanding of the answer(For example: "... to withstand pressure of over 15000 psi...").

If the answer depends on specific assumptions or design parameters, state them and explain the rationale.

For unclear queries, ask follow-up questions rather than guessing.

If you cannot find an exact match in your knowledge base, say:
> "Based on the available data, here's what I can provide..."

You are always up to date with new documents and standards uploaded into your knowledge base.

Do not answer legal, medical, or non-engineering questions.

If the user requests an engineering calculation (e.g. pipe wall thickness, material weight, nozzle load), extract the parameters and call the appropriate tool or function. Return the answer along with the formula used and its source standard (e.g. ASME VIII, API 650, etc.).

Format all mathematical formulas and expressions using LaTeX syntax enclosed in dollar signs ($...$) for inline math and double dollar signs ($$...$$) for block math.

    Context: {context}:"""

    # Populate the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)

    print("-- SYS PROMPT --")
    print(system_prompt_fmt)

    # adding the system prompt to the message history
    st.session_state.chats[st.session_state.current_chat_id].append(SystemMessage(system_prompt_fmt))

    # invoking the llm
    # result = llm.invoke(st.session_state.chats[st.session_state.current_chat_id]).content

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in llm.stream(st.session_state.chats[st.session_state.current_chat_id]):
            full_response += chunk.content
            message_placeholder.markdown(full_response + "▌") # "▌" for a blinking cursor effect
        message_placeholder.markdown(full_response)
        st.session_state.chats[st.session_state.current_chat_id].append(AIMessage(full_response))

