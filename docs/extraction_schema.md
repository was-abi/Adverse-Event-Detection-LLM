# Extraction Schema for Adverse Event Detection

## Overview
This schema defines the structured output for extracting key entities from clinical notes in the `synthetic_ehr.csv` dataset. It is designed for use with an LLM (e.g., Gemini) to identify diagnoses, medications, symptoms, and adverse events. The goal is to enable precise data processing for ML workflows focused on adverse event detection.

## Extraction Targets
For each clinical note, extract the following:

- **diagnosis** (string): The patient's primary condition or diagnosis.  
  Example: "Hypertension"  
  If not mentioned, use `null`.

- **medications** (list of strings): All medications prescribed or mentioned.  
  Example: `["Aspirin", "Metformin"]`  
  If none, use empty list `[]`.

- **symptoms_side_effects** (list of strings): Reported symptoms or side effects.  
  Example: `["rash", "swelling", "chest pain", "dizziness"]`  
  If none, use empty list `[]`.

- **adverse_event** (object): Indicates if an adverse event is present and provides a description.  
  - **present** (boolean): `true` if a symptom is linked to a medication (e.g., via "post [med]" or similar causation phrasing); otherwise `false`.  
  - **description** (string): Brief summary of the event, including linkage if applicable.  
    Example: `{ "present": true, "description": "Rash and swelling post Aspirin" }`  
    If not present, `{ "present": false, "description": "" }`.

## Edge Cases
- **No adverse indicators**: Set `adverse_event.present` to `false` and empty lists where appropriate (e.g., "Normal exam, no adverse reaction noted.").
- **Multiple entities**: List all (e.g., multiple meds or symptoms).
- **Ambiguous causation**: Infer based on phrasing like "after receiving [med]" or "post [med]"; default to `false` if unclear.
- **Missing info**: Use `null` for diagnosis or empty structures for lists/objects.

## Example Extractions
### Sample Note 1
Note: "Patient with hypertension prescribed Aspirin. Developed rash and swelling post Aspirin."

```json
{
  "diagnosis": "hypertension",
  "medications": ["Aspirin"],
  "symptoms_side_effects": ["rash", "swelling"],
  "adverse_event": {
    "present": true,
    "description": "Rash and swelling post Aspirin"
  }
}
```

### Sample Note 2
Note: "Normal exam, no adverse reaction noted."

```json
{
  "diagnosis": null,
  "medications": [],
  "symptoms_side_effects": [],
  "adverse_event": {
    "present": false,
    "description": ""
  }
}
```

### Sample Note 3
Note: "Complains of chest pain after receiving Albuterol. PT with Asthma."

```json
{
  "diagnosis": "Asthma",
  "medications": ["Albuterol"],
  "symptoms_side_effects": ["chest pain"],
  "adverse_event": {
    "present": true,
    "description": "Chest pain after receiving Albuterol"
  }
}
```

## Integration Notes
- Use this schema in prompts for LLMs to enforce JSON output.
- Validate extractions against ground truth (e.g., "adverse_event" column in CSV).
- For batch processing, load into pandas DataFrame and merge with original data.

See `src/extraction.py` for Python implementation and prompt template.
