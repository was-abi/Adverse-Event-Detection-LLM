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

## To-Do Next
- [ ] Set up environment and requirements.txt
- [ ] Simulate/gather sample EHR data
- [ ] Quick EDA notebook
- [ ] Design LLM prompt/prototype Gemini API call
- [ ] Train baseline tabular ML model