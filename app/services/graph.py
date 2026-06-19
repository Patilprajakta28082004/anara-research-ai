"""
Corrective RAG (CRAG) pipeline, built with LangGraph.

Flow:
  retrieve -> grade_documents -> (web_search if irrelevant) -> generate

The LLM is built from whichever provider keys are actually configured
(Gemini -> Groq -> OpenAI, in that fallback order) so the demo still runs
if you only have one free-tier key (Groq's free tier is the easiest to
get for an interview demo).

Grading deliberately avoids forced JSON/tool-call structured output: some
free-tier models (e.g. Groq's llama-3.3-70b-versatile) are unreliable at
strictly following a JSON schema and will occasionally emit garbage. Asking
for a single plain-text word and parsing it ourselves is far more robust.
"""
from typing import Any, Dict, List, TypedDict

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

from app.config import settings
from app.db.chroma import get_chroma_db


def build_llm() -> BaseChatModel:
    """Build a chat model with fallbacks, using only the providers that have keys set."""
    candidates: List[BaseChatModel] = []

    if settings.GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI

        candidates.append(
            ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_retries=2)
        )
    if settings.GROQ_API_KEY:
        from langchain_groq import ChatGroq

        candidates.append(
            ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=2)
        )
    if settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        candidates.append(
            ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2)
        )

    if not candidates:
        raise HTTPException(
            status_code=500,
            detail=(
                "No LLM provider API key is configured. Set at least one of "
                "GOOGLE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY in your .env file."
            ),
        )

    primary, *fallbacks = candidates
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


# ---------------------------------------------------------------------------
# 1. Graph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    question: str
    documents: List[Document]
    web_fallback: bool
    generation: str
    sources: List[dict]


GRADE_PROMPT = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question.

Retrieved document:
{context}

User question: {question}

If the document contains keywords or semantic meaning related to the question, it is relevant.
Answer with exactly one word, lowercase, no punctuation, no explanation: yes or no""",
    input_variables=["context", "question"],
)

GENERATE_PROMPT = PromptTemplate(
    template="""You are an assistant for research tasks. Use the retrieved context below to
answer the question. If the answer isn't in the context, say you don't know.
Keep the answer concise and academic.

Question: {question}

Context: {context}

Answer:""",
    input_variables=["question", "context"],
)


# ---------------------------------------------------------------------------
# 2. Nodes
# ---------------------------------------------------------------------------
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    db = get_chroma_db()
    docs = db.similarity_search(question, k=4) or []
    return {"documents": docs, "question": question}


def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    llm = build_llm()
    grade_chain = GRADE_PROMPT | llm

    filtered_docs = []
    web_fallback = False

    if not documents:
        web_fallback = True
    else:
        for d in documents:
            raw = grade_chain.invoke({"question": question, "context": d.page_content})
            answer = (raw.content or "").strip().lower()
            is_relevant = answer.startswith("yes")
            if is_relevant:
                print("---GRADE: RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: NOT RELEVANT---")
                web_fallback = True

    return {"documents": filtered_docs, "question": question, "web_fallback": web_fallback}


def web_search_fallback(state: GraphState) -> Dict[str, Any]:
    """Stand-in for a real web search call when local context is insufficient.

    Swap this for a real provider (Tavily, Serper, Bing) in production -
    kept as a clearly-labelled mock so the demo doesn't need another API key.
    """
    print("---WEB SEARCH FALLBACK (MOCK)---")
    question = state["question"]
    documents = state["documents"]

    mock_doc = Document(
        page_content=(
            "No sufficiently relevant local context was found for this question. "
            "In production this node would call a live web search API."
        ),
        metadata={"source": "web-fallback (mock)"},
    )
    documents.append(mock_doc)
    return {"documents": documents, "question": question}


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(d.page_content for d in documents)
    sources = [d.metadata for d in documents]

    llm = build_llm()
    chain = GENERATE_PROMPT | llm
    res = chain.invoke({"question": question, "context": context})

    return {"generation": res.content, "sources": sources}


def decide_to_generate(state: GraphState) -> str:
    print("---DECIDE TO GENERATE---")
    return "web_search" if state["web_fallback"] else "generate"


# ---------------------------------------------------------------------------
# 3. Build graph
# ---------------------------------------------------------------------------
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search_fallback)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"web_search": "web_search", "generate": "generate"},
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

crag_app = workflow.compile()


def run_crag_pipeline(query: str) -> Dict[str, Any]:
    """Executes the CRAG pipeline graph for a single question."""
    inputs = {"question": query}
    for output in crag_app.stream(inputs):
        for key, value in output.items():
            print(f"Finished node: {key}")
            if "generation" in value:
                return {"answer": value["generation"], "sources": value.get("sources", [])}

    return {"answer": "Sorry, I couldn't generate an answer.", "sources": []}
