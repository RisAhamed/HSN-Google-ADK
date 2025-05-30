
import os
import asyncio
from google.adk.agents import Agent

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
import re
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Create a logger for this module
from pathlib import Path


try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMER_AVAILABLE = True
    logger.info("Successfully imported sentence-transformers and torch.")
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logger.warning("sentence-transformers or torch not found. HSN suggestion feature will be disabled.")
    SentenceTransformer, util, torch = None, None, None


data_path = Path(__file__).parent / "HSN_data.xlsx" 

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY 

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
MODEL_GEMINI_2_0_FLASH = "gemini-2.5-flash-preview-05-20"

hsn_master_df = pd.DataFrame()
hsn_codes_set = set()
hsn_to_description = {}
description_embeddings = None 
description_list_for_embedding = [] 
suggestion_model = None
VALID_HSN_LENGTHS = {2, 4, 6, 8} 

def load_hsn_data_and_prepare_embeddings():
    global hsn_master_df, hsn_codes_set, hsn_to_description, description_embeddings, suggestion_model, description_list_for_embedding, VALID_HSN_LENGTHS
    try:
        logger.info(f"Attempting to load HSN data from: {data_path.resolve()}")
        if not data_path.exists():
            logger.error(f"CRITICAL ERROR: The HSN data file '{data_path.resolve()}' was not found.")
            return

        df = pd.read_excel(data_path, dtype={'HSNCode': str, 'Description': str})
        logger.info(f"Excel file read. Columns: {df.columns.tolist()}")
        
        required_cols = {'HSNCode': False, 'Description': False}
        temp_renames = {}
        for col in df.columns:
            col_stripped = str(col).strip()
            if col_stripped.upper() == 'HSNCODE' and not required_cols['HSNCode']:
                if col != 'HSNCode': temp_renames[col] = 'HSNCode'
                required_cols['HSNCode'] = True
            elif col_stripped.upper() == 'DESCRIPTION' and not required_cols['Description']:
                if col != 'Description': temp_renames[col] = 'Description'
                required_cols['Description'] = True
        
        if temp_renames:
            df.rename(columns=temp_renames, inplace=True)
            logger.info(f"Renamed columns: {temp_renames}")

        if not required_cols['HSNCode']:
            logger.error(f"'HSNCode' column (or variant) not found in {data_path}. Cannot proceed.")
            return
        if not required_cols['Description']:
            logger.warning(f"'Description' column (or variant) not found. Suggestions might be impaired or descriptions unavailable.")
            df['Description'] = "Description not provided" 

        df['HSNCode'] = df['HSNCode'].astype(str).str.strip()
        df['Description'] = df['Description'].astype(str).str.strip()
        df.dropna(subset=['HSNCode'], inplace=True) # HSNCode is critical
        df['Description'].fillna("Description not provided", inplace=True) # Fill NaN descriptions
        df = df[df['HSNCode'] != ''] 

        hsn_master_df = df.copy()
        hsn_codes_set = set(df['HSNCode'].unique())
        hsn_to_description = pd.Series(df.Description.values, index=df.HSNCode).to_dict()
        logger.info(f"HSN validation data loaded: {len(hsn_codes_set)} unique HSN codes.")
        
       

        if SENTENCE_TRANSFORMER_AVAILABLE:
            logger.info("Preparing description embeddings for suggestions...")
            suggestion_model = SentenceTransformer('all-MiniLM-L6-v2')
            description_list_for_embedding = [
                {'hsn_code': row['HSNCode'], 'description': row['Description']}
                for index, row in hsn_master_df.iterrows() if row['Description'] and row['Description'] != "Description not provided"
            ]
            if description_list_for_embedding:
                descriptions_to_embed = [item['description'] for item in description_list_for_embedding]
                description_embeddings = suggestion_model.encode(descriptions_to_embed, convert_to_tensor=True)
                logger.info(f"Encoded {len(description_list_for_embedding)} descriptions into embeddings.")
            else:
                logger.warning("No valid descriptions found to embed for suggestions.")
        else:
            logger.warning("SentenceTransformer library not available. HSN suggestion by description will not work.")

    except Exception as e:
        logger.error(f"Major error during HSN data loading or embedding: {e}", exc_info=True)
        hsn_master_df = pd.DataFrame(); hsn_codes_set = set(); hsn_to_description = {}
        description_embeddings = None; description_list_for_embedding = []

load_hsn_data_and_prepare_embeddings()

def comprehensive_hsn_validation(code: str) -> dict:
    """
    Validates an HSN code for format, existence, and hierarchy.
    Returns a dictionary with validation status and details.
    """
    code_str = str(code).strip()
    result = {"hsn_code": code_str, "valid": False} # Default to invalid

    if not hsn_codes_set:
        logger.warning("comprehensive_hsn_validation: hsn_codes_set is empty.")
        result["error_message"] = "HSN master data for validation not loaded. Cannot validate."
        result["validation_summary"] = "Data Unloaded"
        return result


    if not code_str.isdigit():
        result["error_message"] = "Invalid format: HSN code must be numeric."
        result["format_status"] = "Error: Not numeric"
        result["validation_summary"] = "Invalid Format (Non-Numeric)"
        return result
    result["format_status"] = "OK (Numeric)"

   
    if len(code_str) not in VALID_HSN_LENGTHS:
        result["error_message"] = f"Invalid format: HSN code length '{len(code_str)}' is not standard. Expected lengths: {VALID_HSN_LENGTHS}."
        result["format_status"] = f"Error: Invalid length ({len(code_str)})"
        result["validation_summary"] = "Invalid Format (Length)"
        return result
    result["format_status"] = f"OK (Numeric, Length {len(code_str)})"


    if code_str not in hsn_codes_set:
        result["error_message"] = f"HSN code {code_str} not found in the master dataset."
        result["existence_status"] = "Not Found"
        result["validation_summary"] = "Not Found"
        return result
    result["existence_status"] = "Found"
    result["description"] = hsn_to_description.get(code_str, "Description not found in map.")

    
    result["hierarchical_status"] = "Not Applicable" 
    missing_parents = []
    if len(code_str) > min(VALID_HSN_LENGTHS): # Only check for codes longer than the base level (e.g. 2)
        current_parent = code_str
        # Iterate backwards, checking parent at each valid length tier
        # E.g., for 01011010 (len 8), parents are 010110 (len 6), 0101 (len 4), 01 (len 2)
        possible_parent_lengths = sorted([l for l in VALID_HSN_LENGTHS if l < len(code_str)], reverse=True)
        
        if possible_parent_lengths: 
            all_parents_found = True
            for parent_len in possible_parent_lengths:
                parent_code_to_check = current_parent[:parent_len]
                if parent_code_to_check not in hsn_codes_set:
                    missing_parents.append(parent_code_to_check)
                    all_parents_found = False
            
            if all_parents_found:
                result["hierarchical_status"] = "OK (All parent codes exist)"
            else:
                result["hierarchical_status"] = f"Warning: Missing parent codes: {', '.join(missing_parents)}"
        else: 
            result["hierarchical_status"] = "OK (Base level code, no parents to check)"


    result["valid"] = True 
    result["status"] = "success" 
    result["message"] = f"HSN code {code_str} is valid. Description: {result['description']}."
    if missing_parents:
        result["message"] += f" Hierarchical check: {result['hierarchical_status']}."
    result["validation_summary"] = "Valid"
    if missing_parents:
       result["validation_summary"] += " (with parent warnings)"
    
    return result


def suggest_hsn_from_description(product_description: str, top_n: int = 3) -> dict:
   
    if not SENTENCE_TRANSFORMER_AVAILABLE or suggestion_model is None or description_embeddings is None:
        logger.warning("suggest_hsn_from_description: Prerequisites for suggestion not met.")
        return {"status": "error", "error_message": "HSN suggestion service is currently unavailable."}
    if not description_list_for_embedding:
         logger.warning("suggest_hsn_from_description: No descriptions were embedded.")
         return {"status": "success", "suggestions": [], "message": "No HSN data available for suggestions."}

    try:
        query_embedding = suggestion_model.encode(product_description, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, description_embeddings)[0]
        k = min(top_n, len(cos_scores))
        if k == 0 :
             return {"status": "success", "suggestions": [], "message": "No HSN data to compare against."}

        top_results = torch.topk(cos_scores, k=k)
        suggestions = []
        for i in range(len(top_results[0])):
            score = top_results[0][i].item()
            corpus_index = top_results[1][i].item()
            original_item = description_list_for_embedding[corpus_index]
            suggestions.append({
                "hsn_code": original_item['hsn_code'],
                "description": original_item['description'],
                "similarity_score": round(score, 4)
            })
        
        if not suggestions:
            return {"status": "success", "suggestions": [], "message": f"No relevant HSN codes found for '{product_description}'."}
        return {"status": "success", "suggestions": suggestions, "message": f"Found {len(suggestions)} suggestions for '{product_description}'."}
    except Exception as e:
        logger.error(f"Error during HSN suggestion for '{product_description}': {e}", exc_info=True)
        return {"status": "error", "error_message": f"An error occurred while suggesting HSN codes: {str(e)}"}


hsnagent = Agent(
    model=MODEL_GEMINI_2_0_FLASH,
    name="hsn_agent",
    description="Validates HSN codes (format, existence, hierarchy) and suggests HSN codes based on product descriptions.",
    instruction="""You are an intelligent assistant specializing in Harmonized System Nomenclature (HSN) codes.
You have two primary functions:
1.  **Validate HSN Codes**: If a user provides an HSN code or asks for validation.
    *   Extract the HSN code.
    *   Use the `comprehensive_hsn_validation` tool with the extracted code.
    *   Based on the tool's output:
        *   If `valid` is true: State its validity, provide the `description`, and mention the `format_status` and `hierarchical_status`.
        *   If `valid` is false: State it's invalid and provide the `error_message` and `validation_summary`.
    *   Example: User: "check 01011010" -> You: Use `comprehensive_hsn_validation` with code="01011010". 
    Response shouldstructured in the following order
    1. format, 
    2. existence, 
    3. description, 4.hierarchy check .

2.  **Suggest HSN Codes from Description**: If a user provides a product/service description and asks for its HSN code.
    *   Extract the product/service description.
    *   Use the `suggest_hsn_from_description` tool.
    *   Present the suggested HSN codes (with descriptions and similarity scores). If no suggestions, inform the user.
    *   Example: User: "What's the HSN for live horses?" -> You: Use `suggest_hsn_from_description` with product_description="live horses".

If the user's query is ambiguous, ask for clarification. If the suggestion service is unavailable, inform the user.
When reporting validation results, be thorough. For a valid code, confirm its description and mention the status of format and hierarchical checks (e.g., "Format is OK. Hierarchical structure is OK." or "Format is OK.
Hierarchical check: Warning, missing parent code XXXX."). For an invalid code, clearly state the reason based on the `validation_summary` or `error_message` from the tool.
""",
    tools=[suggest_hsn_from_description,comprehensive_hsn_validation]
)

session_service = InMemorySessionService()
_default_session = session_service.create_session(
    app_name="hsn_app_prog", user_id="prog_user", session_id="prog_sess"
)
runner = Runner(agent=hsnagent, app_name="hsn_app_prog", session_service=session_service)

async def chat(query: str, runner_instance: Runner, user_id: str, session_id: str):
    logger.info(f"\n>>> User Query ({user_id}): {query}")
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."
    print(f"<<< {hsnagent.name} Response: ", end="", flush=True)
    async for event in runner_instance.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_llm_response_chunk() and event.content and event.content.parts:
            chunk_text = event.content.parts[0].text
            print(chunk_text, end="", flush=True)
            if final_response_text == "Agent did not produce a final response.": final_response_text = "" 
            final_response_text += chunk_text
        elif event.is_final_response():
            if event.content and event.content.parts and final_response_text == "Agent did not produce a final response.":
                 final_response_text = event.content.parts[0].text
                 print(final_response_text, end="")
            elif final_response_text == "Agent did not produce a final response.":
                if event.error_message: final_response_text = f"(Agent error: {event.error_message})"
                elif event.actions and event.actions.escalate: final_response_text = f"(Agent escalated: {event.error_message or 'No specific message.'})"
                print(final_response_text, end="")
            break 
    print()
