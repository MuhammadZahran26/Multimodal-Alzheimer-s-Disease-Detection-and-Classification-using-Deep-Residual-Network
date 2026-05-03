# Load all necessary modules
import os
from dotenv import load_dotenv
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# old imports commented out
# from langchain_classic.chains import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

# NEW imports for updated langchain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever


class MedicalChatbot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")

        self.user_sessions = {}
        self.db_path = "vectorstore/"
        self.similarity_threshold = 1.2

        self.setup_vector_db()
        self.setup_llm()
        self.setup_chain()

        print("✅ Medibot initialized")

    # --------------------------------------------------
    # Vector DB
    # --------------------------------------------------
    def setup_vector_db(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        self.db = FAISS.load_local(
            self.db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.db.as_retriever(search_kwargs={"k": 4})

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------
    def setup_llm(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.3,
            max_output_tokens=1024
        )

    # --------------------------------------------------
    # Embedding-based scope guard
    # --------------------------------------------------
    def is_query_in_scope(self, query: str) -> bool:
        query_vec = self.embeddings.embed_query(query)

        docs_and_scores = self.db.similarity_search_with_score(
            query, k=1
        )

        if not docs_and_scores:
            return False

        _, score = docs_and_scores[0]

        # FAISS score: lower = more similar
        return score <= self.similarity_threshold

    # --------------------------------------------------
    # RAG Chain
    # --------------------------------------------------
    def setup_chain(self):
        reformulation_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Rewrite the user's question so it is standalone.\n"
             "- Keep it short\n"
             "- Do not answer\n"
             "- Do not add new info"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            reformulation_prompt
        )

        qa_system_prompt = (
            "You are Medibot, a medical assistant specialized ONLY in Alzheimer’s disease "
            "and related dementias.\n\n"
            "RULES (STRICT):\n"
            "- Answer ONLY using the provided context.\n"
            "- If the question is not about Alzheimer’s disease or dementia, politely refuse.\n"
            "- If the context does not contain the answer, say you do not know.\n"
            "- Do NOT speculate or invent information.\n\n"
            "MEDICAL SAFETY:\n"
            "- If medications, treatments, or therapies are mentioned, end with:\n"
            "\"⚠️ Disclaimer: This information is for educational purposes only. "
            "Always consult a qualified doctor before starting or changing any treatment.\"\n\n"
            "VERBOSITY:\n"
            "- Be concise by default.\n"
            "- Be detailed ONLY if explicitly asked.\n\n"
            "FORMAT:\n"
            "- Use Markdown for formatting (bold, bullet points).\n"
            "- Use bullet points where helpful.\n"
            "- Avoid unnecessary medical jargon.\n\n"
            "Context:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(
            history_aware_retriever,
            qa_chain
        )

    # --------------------------------------------------
    # Memory helpers
    # --------------------------------------------------
    def summarize_messages(self, summary, messages):
        text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}"
            for m in messages
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Summarize the Alzheimer-related conversation.\n"
             "- 3 to 5 bullet points\n"
             "- Focus on medical facts"),
            ("human", text)
        ])

        response = (prompt | self.llm).invoke({})
        return response.content.strip()

    # --------------------------------------------------
    # Main chat method
    # --------------------------------------------------
    def get_response(self, user_input: str, session_id: str) -> str:

        # 🔒 Embedding-based guard (natural)
        if not self.is_query_in_scope(user_input):
            return (
                "I can help with questions related to Alzheimer’s disease and dementia. "
                "If your question is about that, feel free to ask."
            )

        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = {
                "summary": "",
                "messages": []
            }

        session = self.user_sessions[session_id]

        history = []
        if session["summary"]:
            history.append(AIMessage(content=session["summary"]))
        history.extend(session["messages"])

        result = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": history
        })

        answer = result["answer"]

        session["messages"].append(HumanMessage(content=user_input))
        session["messages"].append(AIMessage(content=answer))

        if len(session["messages"]) >= 8:
            session["summary"] = self.summarize_messages(
                session["summary"], session["messages"]
            )
            session["messages"] = session["messages"][-2:]

        return answer
