from neo4j import GraphDatabase
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .llm_factory import get_llm
import os
import pandas as pd
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jChatMemory:
    """Manages chat history and session persistence in the Neo4j Knowledge Graph."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.pwd = os.getenv("NEO4J_PASSWORD")
        if not all([self.uri, self.user, self.pwd]):
            raise ValueError("Neo4j credentials missing for ChatMemory.")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))

    def add_message(self, role: str, content: str):
        """Creates a ChatMessage node and links it to the Session."""
        with self.driver.session() as session:
            # 1. Merge the Session node
            # 2. Create the Message node
            # 3. Link Message to Session
            # 4. (Optional) Chain to the previous message for thread ordering
            cypher = """
            MERGE (s:UserSession {id: $session_id})
            ON CREATE SET s.started_at = datetime()
            
            CREATE (m:ChatMessage {
                id: $msg_id,
                role: $role,
                content: $content,
                timestamp: datetime()
            })
            MERGE (s)-[:HAS_MESSAGE]->(m)
            
            WITH s, m
            MATCH (s)-[:HAS_MESSAGE]->(prev:ChatMessage)
            WHERE prev <> m AND NOT (prev)-[:NEXT]->()
            MERGE (prev)-[:NEXT]->(m)
            """
            session.run(cypher, session_id=self.session_id, role=role, content=content, msg_id=str(uuid.uuid4()))

    def get_history(self, limit: int = 5) -> str:
        """Retrieves the last N messages to provide as context to the AI."""
        with self.driver.session() as session:
            cypher = """
            MATCH (s:UserSession {id: $session_id})-[:HAS_MESSAGE]->(m:ChatMessage)
            RETURN m.role as role, m.content as content, m.timestamp as ts
            ORDER BY m.timestamp DESC LIMIT $limit
            """
            results = session.run(cypher, session_id=self.session_id, limit=limit)
            history = [f"{r['role'].upper()}: {r['content']}" for r in results]
            history.reverse() # Keep chronological order for the prompt
            return "\n".join(history) if history else "No previous conversation."

    def close(self):
        self.driver.close()

def get_domain_context(department: str = None, service_type: str = None, **kwargs) -> str:
    """
    Expert Hybrid Retrieval: 
    1. Cypher lookup against Neo4j Graph (Structured Audit Rules).
    2. Vector Search against DocumentChunk nodes (Unstructured PDF context).
    """
    if not all([URI, USERNAME, PASSWORD]):
        return "Neo4j configuration missing in .env."
        
    context_parts = ["CITY KNOWLEDGE MAP (Expert Hybrid Context):"]
    
    try:
        driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        
        # --- PART 1: Cypher Graph Search (Structured Audit Rules) ---
        if department:
            with driver.session() as session:
                cypher = """
                MATCH (d:Department)-[:MONITORED_IN]->(a:AuditTopic)
                WHERE d.name =~ $dept_regex
                RETURN d.name AS dept, a.name AS topic, a.objective AS objective
                """
                # Use regex for flexible department matching
                dept_regex = f"(?i).*{department}.*"
                results = session.run(cypher, dept_regex=dept_regex)
                for record in results:
                    context_parts.append(f"- Audit Focus: {record['topic']} (Goal: {record['objective']})")

        # --- PART 2: Vector Search with Neo4j Natively ---
        embeddings_service = VertexAIEmbeddings(model_name="text-embedding-004")
        search_query = f"Staffing, budget, and performance impacts for department {department or 'Dallas 311'}"
        query_embedding = embeddings_service.embed_query(search_query)
        
        with driver.session() as session:
            # 0. Fetch Expert Human Wisdom (The Golden Source)
            expert_cypher = "MATCH (w:ExpertWisdom) RETURN w.topic AS topic, w.content AS content"
            expert_results = session.run(expert_cypher)
            expert_wisdom = list(expert_results)
            if expert_wisdom:
                context_parts.append("\n- GOLDEN HUMAN WISDOM (Team Expert Interpretations):")
                for w in expert_wisdom:
                    context_parts.append(f"  * {w['topic']}: {w['content']}")

            # 1. Fetch Top 1 Insight Source (Explorer Agent real-time search)
            insight_cypher = """
            MATCH (c:DocumentChunk {type: 'Insight'})
            WHERE c.service =~ $service_regex OR c.content CONTAINS $service_type
            RETURN c.content AS content, c.timestamp AS ts
            ORDER BY ts DESC LIMIT 1
            """
            service_regex = f"(?i).*{service_type or department or ''}.*"
            insight_results = session.run(insight_cypher, service_regex=service_regex, service_type=service_type or "")
            insight_record = insight_results.single()
            if insight_record:
                context_parts.append("\n- WEB-OPTIMIZED INSIGHTS (Explorer Agent):")
                context_parts.append(f"  [Updated: {insight_record['ts']}] {insight_record['content'][:500]}...")

            # 2. Fetch Top 1 Golden Source (High priority)
            golden_cypher = """
            CALL db.index.vector.queryNodes('chunk_embeddings', 5, $emb) 
            YIELD node, score 
            WHERE node.type = 'Golden'
            RETURN node.content AS content, node.source AS source
            LIMIT 1
            """
            golden_results = session.run(golden_cypher, emb=query_embedding)
            golden_record = golden_results.single()
            if golden_record:
                context_parts.append("\n- GOLDEN BUSINESS RULES (High Priority):")
                context_parts.append(f"  [Ref: {golden_record['source']}] {golden_record['content'][:500]}...")

            # 2. Fetch Top 2 Report Sources (Contextual)
            report_cypher = """
            CALL db.index.vector.queryNodes('chunk_embeddings', 5, $emb) 
            YIELD node, score 
            WHERE node.type = 'Report'
            RETURN node.content AS content, node.source AS source
            LIMIT 2
            """
            report_results = session.run(report_cypher, emb=query_embedding)
            reports = list(report_results)
            if reports:
                context_parts.append("\n- SUPPLEMENTAL REPORT CONTEXT:")
                for record in reports:
                    context_parts.append(f"  [Source: {record['source']}] {record['content'][:300]}...")

            # 3. Fetch External Factors (Seasonality / Weather Backlogs)
            factor_cypher = """
            MATCH (s:Service {name: $service_type})-[:AFFECTED_BY]->(f:ExternalFactor)
            RETURN f.name AS factor, f.impact AS impact
            """
            factor_results = session.run(factor_cypher, service_type=service_type or "")
            factors = list(factor_results)
            if factors:
                context_parts.append("\n- OPERATIONAL BACKLOGS (External Factors):")
                for record in factors:
                    context_parts.append(f"  * {record['factor']}: {record['impact']}")

        driver.close()

        if len(context_parts) == 1:
            return f"Standard Dallas 311 procedures apply for {department}."
            
        return "\n".join(context_parts)
    except Exception as e:
        return f"Hybrid Context Error: {e}"

def create_explainability_chain(provider: str = None):
    """
    Creates an LCEL chain for explaining ML model coefficients.
    
    The chain expects a dictionary with:
    - 'prediction': the numerical prediction result.
    - 'coef_summary': a string or dict representation of feature importance.
    - 'department': (Optional) the city department for KG lookup.
    - 'district': (Optional) city council district.
    """
    
    llm = get_llm(provider=provider, temperature=0.2)
    
    # Define the prompt template for the model
    prompt = ChatPromptTemplate.from_template("""
    You are the DALLAS 311 STRATEGIC ADVISOR (Opus-Level Reasoner).
    You don't just report status; you interpret the city's operational rhythm and provide expert-level guidance.
    
    ---
    SUB-GRAPH CONTEXT (from Knowledge Graph):
    {domain_context}
    
    ---
    MODEL RESULT:
    - Prediction: {prediction}
    - Key Influencing Features:
    {coef_summary}
    ---
    
    INSTRUCTIONS (The 4th Cycle Standard):
    1. EXPLAIN the resolution path based on current heuristics.
    2. IDENTIFY 'invisible' bottlenecks (look at Supplemental Context and External Factors).
    3. PROVIDE a 'Pro-Tip' that only an expert city official would know to speed up the case.
    
    Translate technical coefficients into operational reality. Be nuanced and authoritative.
    
    STRATEGIC ANALYSIS:
    """)
    
    # The LCEL Chain: input -> prompt -> model -> string output
    chain = (
        {
            "prediction": lambda x: x["prediction"],
            "coef_summary": lambda x: x["coef_summary"],
            "domain_context": lambda x: get_domain_context(
                department=x.get("department"), 
                district=x.get("district")
            )
        } 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def format_coef_summary(df_coef: pd.DataFrame, top_n: int = 5) -> str:
    """
    Helper to convert the RegularizationAgent's coefficient DataFrame 
    into a clean string for the LLM prompt.
    """
    if df_coef.empty:
        return "No coefficient data available."
    
    # Sort by absolute magnitude to get most important features
    df_sorted = df_coef.assign(abs_coef=df_coef['Coefficient'].abs()).sort_values(by='abs_coef', ascending=False)
    top_features = df_sorted.head(top_n)
    
    summary = ""
    for _, row in top_features.iterrows():
        direction = "Increases delay" if row['Coefficient'] > 0 else "Decreases delay"
        summary += f"- {row['Feature']}: {row['Coefficient']:.4f} ({direction})\n"
        
    return summary

if __name__ == "__main__":
    # --- PROTOTYPE: RAG Chat with Graph Memory ---
    print("--- TESTING CONVERSATIONAL RAG (+Neo4j Memory) ---")
    session_id = "test_user_777"
    memory = Neo4jChatMemory(session_id)
    
    # 1. Simulate a Human Question
    user_query = "What is the primary concern for Code Compliance right now?"
    print(f"USER: {user_query}")
    
    # 2. Get Context (Facts + History)
    facts = get_domain_context(department="Code Compliance")
    history = memory.get_history(limit=3)
    
    # 3. Use LLM to Answer
    llm = get_llm(temperature=0.2)
    chat_prompt = ChatPromptTemplate.from_template("""
        You are the Dallas 311 Assistant. Use the context and history to answer.
        
        CITY FACTS: {facts}
        CONVERSATION HISTORY: {history}
        HUMAN: {query}
        ASSISTANT:
    """)
    
    chain = chat_prompt | llm | StrOutputParser()
    answer = chain.invoke({"facts": facts, "history": history, "query": user_query})
    
    # 4. Save to Memory
    memory.add_message("human", user_query)
    memory.add_message("ai", answer)
    
    print(f"\nAI: {answer}")
    print("\n--- Memory Saved to Neo4j ---")
    memory.close()
