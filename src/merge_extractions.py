import pandas as pd
import json
import ast  # For safe dict parsing

def merge_extractions():
    """
    Merge llm_extractions.csv with synthetic_ehr.csv and save combined dataset.
    """
    # Load original EHR data
    df_ehr = pd.read_csv('data/synthetic_ehr.csv')
    
    # Load LLM extractions
    df_llm = pd.read_csv('data/llm_extractions.csv')
    
    # Parse adverse_event column if it's a string representation of dict
    def parse_adverse_event(value):
        if pd.isna(value) or value == '':
            return {'present': False, 'description': 'No Adverse Event Detected'}
        try:
            # Try json.loads first
            return json.loads(value)
        except:
            # Fallback to ast.literal_eval for Python dict string
            try:
                return ast.literal_eval(value)
            except:
                return {'present': False, 'description': 'Parsing Failed'}
    
    # Apply parsing to adverse_event column
    adverse_parsed = df_llm['adverse_event'].apply(parse_adverse_event)
    
    # Flatten the parsed adverse_event into separate columns
    df_llm['adverse_event_present'] = adverse_parsed.apply(lambda x: x['present'])
    df_llm['adverse_event_description'] = adverse_parsed.apply(lambda x: x['description'])
    
    # Drop the original adverse_event column
    df_llm = df_llm.drop('adverse_event', axis=1)
    
    # Concat EHR and LLM data on axis=1 (assuming same row order)
    df_combined = pd.concat([df_ehr, df_llm], axis=1)
    
    # Save the combined dataset
    df_combined.to_csv('data/combined_ehr.csv', index=False)
    print("Merge complete. Combined data saved to data/combined_ehr.csv")
    print(f"Shape: {df_combined.shape}")
    print(df_combined.head())
    
    # Save mini dataset (first 12 rows with successful extractions)
    mini_df = df_combined.head(12)
    mini_df.to_csv('data/mini_combined_ehr_llm.csv', index=False)
    print("Mini dataset saved to data/mini_combined_ehr_llm.csv")
    
    return df_combined

if __name__ == "__main__":
    merge_extractions()
