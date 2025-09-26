"""
Extraction module for adverse event detection from clinical notes.
Defines the schema template and LLM prompt for structured extraction.
"""

import json
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Structured extraction template (default values)
extraction_template = {
    "diagnosis": None,  # str or None
    "medications": [],  # list of str
    "symptoms_side_effects": [],  # list of str
    "adverse_event": {
        "present": False,  # bool
        "description": ""   # str
    }
}

def parse_llm_output(llm_response: str, template: dict = extraction_template) -> dict:
    """
    Parse JSON output from LLM and merge with template defaults.
    
    Args:
        llm_response (str): Raw JSON string from LLM.
        template (dict): Default schema template.
    
    Returns:
        dict: Merged extraction dict.
    """
    # Extract JSON block if response has extra text
    json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
    if json_match:
        llm_response = json_match.group(0)
    
    try:
        parsed = json.loads(llm_response)
        # Merge with defaults to handle missing fields
        result = {**template, **parsed}
        # Ensure nested adverse_event is merged
        if "adverse_event" in parsed:
            result["adverse_event"] = {**template["adverse_event"], **parsed["adverse_event"]}
        return result
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in LLM response: {llm_response}")

def get_extraction_prompt(note: str) -> str:
    """
    Generate a prompt for the LLM to extract entities from a clinical note.
    
    Args:
        note (str): The clinical note text.
    
    Returns:
        str: Formatted prompt string.
    """
    schema_desc = """
    {
      "diagnosis": "string or null",
      "medications": ["string", ...],
      "symptoms_side_effects": ["string", ...],
      "adverse_event": {
        "present": true/false,
        "description": "brief string or empty"
      }
    }
    """
    
    prompt = f"""You are a clinical extractor. Analyze the following clinical note and respond with ONLY the JSON object matching the schema. No other text, no explanations, no markdown—pure JSON only.

Schema (exact structure required):
{schema_desc}

Instructions:
- Extract precisely from the note.
- diagnosis: string or null if not mentioned.
- medications: array of strings, empty [] if none.
- symptoms_side_effects: array of strings, empty [] if none.
- adverse_event.present: true ONLY if there is explicit evidence of causation between a symptom/side effect and a medication (e.g., "caused by [med]", "due to [med]", "side effect of [med]", "developed [symptom] as a result of [med]"). Temporal proximity alone (e.g., "after [med]") is NOT sufficient unless causation is clearly stated; else false.
- adverse_event.description: concise description if present, else "No Adverse Event Detected".

Clinical Note: {note}

JSON:"""
    
    return prompt

def extract_with_gemini(note: str, model_name: str = "gemini-2.5-flash") -> dict:
    """
    Extract entities from a clinical note using Gemini API.
    
    Args:
        note (str): The clinical note text.
        model_name (str): Gemini model to use (default: gemini-1.5-flash).
    
    Returns:
        dict: Parsed extraction dict.
    
    Raises:
        ValueError: If API call or parsing fails.
    """
    prompt = get_extraction_prompt(note)
    
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    
    if response.text.strip():
        print("Raw Gemini Response:")
        print(response.text)  # Debug: Show raw output
        try:
            return parse_llm_output(response.text.strip())
        except ValueError as e:
            raise ValueError(f"Failed to parse Gemini response: {e}")
    else:
        raise ValueError("Empty response from Gemini")

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('data/synthetic_ehr.csv')
    sample_data = df.head(5)
    print("Extracting with Gemini on 5 samples from synthetic_ehr.csv...")
    for index, row in sample_data.iterrows():
        note = row['note']
        ground_truth = row['adverse_event']
        try:
            extracted = extract_with_gemini(note)
            print(f"\nSample {index}:")
            print(f"Note: {note}")
            print("Extracted Data:")
            print(json.dumps(extracted, indent=2))
            match = ground_truth == extracted['adverse_event']['present']
            print(f"Ground Truth Adverse Event: {ground_truth}, Extracted: {extracted['adverse_event']['present']}, Match: {match}")
        except Exception as e:
            print(f"Error during extraction for sample {index}: {e}")
            print("Ensure GEMINI_API_KEY is set in .env and dependencies are installed.")
