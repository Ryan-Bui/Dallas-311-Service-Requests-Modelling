from google.cloud import spanner
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .llm_factory import get_llm
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
INSTANCE_ID = os.getenv("GCP_INSTANCE_ID")
DATABASE_ID = os.getenv("GCP_DATABASE_ID")
GRAPH_NAME = "Dallas311Graph"
DOC_TABLE = "DocumentChunks"

def get_domain_context(department: str = None, service_type: str = None, **kwargs) -> str:
    """
    Expert Hybrid Retrieval: 
    1. GQL lookup against Spanner Graph (Structured rules).
    2. Vector Search against DocumentChunks (Unstructured PDF context).
    """
    if not PROJECT_ID or not INSTANCE_ID:
        return "GCP Spanner configuration missing in .env."
        
    context_parts = ["CITY KNOWLEDGE MAP (Expert Hybrid Context):"]
    
    try:
        # Initialize Spanner Client
        client = spanner.Client(project=PROJECT_ID)
        instance = client.instance(INSTANCE_ID)
        database = instance.database(DATABASE_ID)

        # --- PART 1: GQL Graph Search (Structured Audit Rules) ---
        if department:
            gql_query = f"""
            GRAPH {GRAPH_NAME}
            MATCH (d:Departments {{Name: @dept}})-[:MonitoredIn]->(a:AuditTopics)
            RETURN d.Name AS dept, a.TopicName AS topic, a.Objective AS objective
            """
            with database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    gql_query, 
                    params={"dept": department},
                    param_types={"dept": spanner.param_types.STRING}
                )
                rows = list(results)
                for row in rows:
                    context_parts.append(f"- Audit Focus: {row[1]} (Goal: {row[2]})")

        # --- PART 2: Vector Search with Golden Prioritization ---
        from langchain_google_vertexai import VertexAIEmbeddings
        embeddings_service = VertexAIEmbeddings(model_name="text-embedding-004")
        
        search_query = f"Staffing, budget, and performance impacts for department {department or 'Dallas 311'}"
        query_embedding = embeddings_service.embed_query(search_query)
        
        # 1. Fetch Top 1 Golden Source (High priority)
        golden_sql = f"""
        SELECT Content, SourceFile 
        FROM {DOC_TABLE} 
        WHERE SourceType = 'Golden'
        ORDER BY COSINE_DISTANCE(Embedding, @emb) 
        LIMIT 1
        """
        
        # 2. Fetch Top 2 Report Sources (Contextual)
        report_sql = f"""
        SELECT Content, SourceFile 
        FROM {DOC_TABLE} 
        WHERE SourceType = 'Report'
        ORDER BY COSINE_DISTANCE(Embedding, @emb) 
        LIMIT 2
        """
        
        with database.snapshot(multi_use=True) as snapshot:
            # Query Golden Source
            golden_results = snapshot.execute_sql(
                golden_sql,
                params={"emb": query_embedding},
                param_types={"emb": spanner.param_types.Array(spanner.param_types.FLOAT64)}
            )
            
            golden_rows = list(golden_results)
            if golden_rows:
                context_parts.append("\n- GOLDEN BUSINESS RULES (High Priority):")
                for row in golden_rows:
                    context_parts.append(f"  [Ref: {row[1]}] {row[0][:500]}...")

            # Query Report Sources
            report_results = snapshot.execute_sql(
                report_sql,
                params={"emb": query_embedding},
                param_types={"emb": spanner.param_types.Array(spanner.param_types.FLOAT64)}
            )
            
            report_rows = list(report_results)
            if report_rows:
                context_parts.append("\n- SUPPLEMENTAL REPORT CONTEXT:")
                for row in report_rows:
                    context_parts.append(f"  [Source: {row[1]}] {row[0][:300]}...")

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
    You are an expert Data Science Assistant for the City of Dallas 311 Service.
    
    A machine learning model has made a prediction regarding a service request. 
    Your task is to explain the "why" behind this prediction using the provided model metadata.
    
    ---
    DOMAIN KNOWLEDGE (from Knowledge Graph):
    {domain_context}
    
    ---
    MODEL RESULT:
    - Prediction: {prediction}
    - Key Influencing Features (Coefficients):
    {coef_summary}
    ---
    
    INSTRUCTIONS:
    - Translate the statistical coefficients into plain English for a city official.
    - IMPORTANT: Explicitly cite the DOMAIN KNOWLEDGE (Audit focus or Report context) to explain if the prediction exceeds city standards or planned audit objectives.
    - If a feature has a high positive coefficient, it increased the predicted value.
    - Be concise but professional. 
    
    EXPLANATION:
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
    # Quick Test with Mock Data
    print("Testing LCEL Explainability Chain...")
    try:
        mock_coef = pd.DataFrame([
            {"Feature": "noise_level", "Coefficient": 0.85},
            {"Feature": "is_weekend", "Coefficient": -0.42},
            {"Feature": "ward_region_4", "Coefficient": 0.12}
        ])
        
        test_chain = create_explainability_chain()
        summary_str = format_coef_summary(mock_coef)
        
        # Invoke the chain
        response = test_chain.invoke({
            "prediction": "12.5 Days to Close",
            "coef_summary": summary_str
        })
        
        print("\n--- LLM Response ---")
        print(response)
        
    except Exception as e:
        print(f"Error during test: {e}")
