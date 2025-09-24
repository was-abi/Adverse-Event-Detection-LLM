# Adverse-Event-Detection-LLM
Goal: “Build a real-time pipeline that uses LLMs (like Gemini 2.5 Flash) to extract relevant details from clinical notes and combines these with structured EHR data to predict or flag potential adverse medical events.”

Motivation: Early detection is critical; doctors log most relevant info as text.

Success Metric: E.g., “High recall for critical events, reasonable API latency.”

# Adverse-Event-Detection-LLM
End-to-end pipeline for real-time adverse event detection in clinical EHR using a combination of Gemini 2.5 Flash and classic ML models.

## Project Outline
- Ingest EHR data (structured and unstructured notes)
- Use Gemini LLM for text extraction/classification
- Fuse with tabular ML model for risk scoring
- Deploy as API with monitoring, CI/CD, and explainability

## Data

Simulated EHR data is generated using the `ehr_data_generator.py` script in the root directory. The output is stored in the `data/` directory.

**What was simulated:** Synthetic clinical notes and structured EHR records, including patient demographics, vital signs, medications, lab results, and free-text clinical narratives that mimic scenarios involving potential adverse events (e.g., drug reactions, procedural complications).

**Data references:** 
- Generator script: `ehr_data_generator.py`
- Sample data: `data/` directory (explore subfolders for raw and processed files)

**Data issues:**
- Synthetic data lacks the full complexity and variability of real EHR data.
- Potential biases in simulation (e.g., over-representation of common events).
- No real patient identifiers; for production use, real data must be de-identified per HIPAA/GDPR standards.
- Limited scale; may require augmentation for robust model training.

## Next Steps
- Design LLM prompts for extracting and classifying adverse events from unstructured clinical notes.
- Integrate Gemini API (e.g., via Google AI SDK) for real-time inference on incoming EHR data.

## To-Do Next
- [ ] Set up environment and requirements.txt
- [ ] Simulate/gather sample EHR data
- [ ] Quick EDA notebook
- [ ] Design LLM prompt/prototype Gemini API call
- [ ] Train baseline tabular ML model
