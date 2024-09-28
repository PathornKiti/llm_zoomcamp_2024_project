import json
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchRetriever
from typing import Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os

#Openai Key
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API Key is not set. Please export OPENAI_API_KEY.")

# Constants
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
ES_URL = 'http://elasticsearch:9200'
INDEX_NAME = 'relationship_consult'

# Initialize embeddings and Elasticsearch client
embeddings = SentenceTransformerEmbeddings(model_name=MODEL_NAME)
es_client = Elasticsearch(ES_URL)

# LLM System Prompt
SYSTEM_PROMPT = '''
You are Saddie a relationship advisor chatbot. Your role is to provide thoughtful, empathetic, and actionable advice to users based on the given context. You will receive a variable `{context}` that provides important information about the user's situation or emotional state. When responding:

1. **Empathy First**: Always approach each situation with understanding, compassion, and without judgment, considering the emotional tone provided in `{context}`.
2. **Tailor Responses**: Use the details in `{context}` to customize your advice. Whether the user is feeling hurt, confused, or happy, adjust your tone and suggestions accordingly.
3. **Balanced Advice**: Provide balanced perspectives, considering both emotional and practical aspects of relationship dynamics. Always factor in the specific details of `{context}`.
4. **Clarity**: Keep responses clear, concise, and free from jargon. Ensure your advice is actionable and suited to the user's situation as described in `{context}`.
5. **Non-Biased**: Avoid taking sides in conflicts; instead, focus on encouraging healthy communication, mutual respect, and personal growth. Be sensitive to any biases or specific issues mentioned in `{context}`.
6. **Emotionally Supportive**: Be positive, uplifting, and sensitive to the emotions involved in the conversation, as described in `{context}`.
7. **Encourage Communication**: When appropriate, remind users of the importance of open and honest communication with their partners, adjusting advice based on their specific needs from `{context}`.
8. **Resource Suggestion**: In cases where external help might be useful (such as therapy or professional consultation), gently suggest these resources, especially if `{context}` suggests a need for deeper intervention.

You should be respectful of all types of relationships and inclusive of different genders, orientations, and cultural backgrounds. Always adapt your tone to the user's emotional state, as indicated by `{context}`, providing comfort and support where necessary.
'''

# Contextualizing prompt for history-aware retriever
CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# --- Functions --- #

def initialize_hybrid_retriever(index_name: str, es_url: str) -> ElasticsearchRetriever:
    """Initializes the hybrid retriever using Elasticsearch and sentence embeddings."""
    def hybrid_query(query: str) -> Dict:
        vector = embeddings.embed_query(query)  # same embeddings as for indexing
        return {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fuzziness": "AUTO",
                            "fields": ["content", "title", "description"],
                            "type": "best_fields",
                            "boost": 0.5,
                        }
                    },
                }
            },
            "knn": {
                "field": "content_vector",
                "query_vector": vector,
                "k": 3,
                "num_candidates": 10000,
                "boost": 0.5,
            },
            "size": 3,
        }

    return ElasticsearchRetriever.from_es_params(
        index_name=index_name,
        body_func=hybrid_query,
        content_field='content',
        url=es_url,
    )


def initialize_llm_chain() -> ChatOpenAI:
    """Initializes the language model."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def initialize_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    """Initializes a chat prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )


def create_rag_chain(llm: ChatOpenAI, retriever: ElasticsearchRetriever) -> any:
    """Creates a Retrieval-Augmented Generation (RAG) chain with the LLM and retriever."""
    prompt = initialize_prompt_template(SYSTEM_PROMPT)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def create_contextual_retriever(llm: ChatOpenAI, retriever: ElasticsearchRetriever) -> any:
    """Creates a history-aware retriever using a contextual prompt."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


def compile_conversational_chain(llm: ChatOpenAI, retriever: ElasticsearchRetriever) -> RunnableWithMessageHistory:
    """Creates a conversational chain with message history."""
    qa_prompt = initialize_prompt_template(SYSTEM_PROMPT)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    history_aware_retriever = create_contextual_retriever(llm, retriever)

    # Create final chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Session management for chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or initializes a session's chat history."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def initialize_conversational_chain(llm: ChatOpenAI, retriever: ElasticsearchRetriever) -> RunnableWithMessageHistory:
    """Initializes a conversational chain with message history."""
    conversational_rag_chain = RunnableWithMessageHistory(
        compile_conversational_chain(llm, retriever),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


# --- Main Compilation Function --- #

def compile_llm_pipeline():
    """Compiles the full LLM pipeline with retrieval, history, and question answering."""
    retriever = initialize_hybrid_retriever(INDEX_NAME, ES_URL)
    llm = initialize_llm_chain()

    # Create the RAG chain with history awareness and return it
    conversational_chain = initialize_conversational_chain(llm, retriever)
    
    return conversational_chain
