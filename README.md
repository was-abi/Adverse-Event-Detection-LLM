# Adverse-Event-Detection-LLM
# Day 1:

Goal: “Build a real-time pipeline that uses LLMs (like Gemini 2.5 Flash) to extract relevant details from clinical notes and combines these with structured EHR data to predict or flag potential adverse medical events.”

Motivation: Early detection is critical; doctors log most relevant info as text.

Success Metric: E.g., “High recall for critical events, reasonable API latency.”

# Day 2:
Day 2: LLM Prompt Engineering Step-by-Step
Goal
Create, test, and iterate prompts that allow Gemini to:

Summarize a doctor’s note for adverse event risk.

Extract structured entities (diagnosis, medication, symptoms, adverse event signals) from messy clinical text.

# Day 3: Prompt Iteration and Documentation
Goal: Refine prompts based on initial testing, address issues like false positives in adverse event detection, and document the process.

Progress:
- Revised the extraction prompt in `src/extraction.py` to require explicit causation (e.g., "caused by", "due to") for adverse_event.present, avoiding over-interpretation of temporal phrases like "after".
- Re-tested on 5 synthetic samples from `data/synthetic_ehr.csv`: Fixed false positive in Sample 1 (now 80% accuracy, balanced precision/recall trade-off noted).
- Updated `notebooks/02_llm_prompt_engineering.ipynb` with test results, revised prompt template, feedback, and recommendations (e.g., add few-shot examples for future iterations).

This iteration improves reliability for the LLM extraction step. Next: Integrate into pipeline.

# Day 4: Tabular ML Model — Baseline Adverse Event Classification
Goal
Build, train, and evaluate a simple ML model (e.g., Logistic Regression, Decision Tree) using only structured EHR data fields.

Document pipeline and results for comparison with LLM-augmented approach.

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

## Prompt Engineering Progress
The LLM prompt engineering for entity extraction has been implemented in `src/extraction.py`, using Gemini 2.5 Flash to extract structured data (diagnosis, medications, symptoms/side effects, adverse_event) from clinical notes.

- **Key Features**: Strict JSON schema enforcement, prompt for precise extraction, parsing with fallback for extra text.
- **Testing**: Evaluated on 5 synthetic samples from `data/synthetic_ehr.csv`. Initial accuracy 80% for adverse event detection. Revised prompt to require explicit causation (e.g., "caused by", "due to") over temporal phrases, fixing false positives but noting trade-offs in synthetic data labeling.
- **Documentation**: Full details, examples, issues, and iteration results in `notebooks/02_llm_prompt_engineering.ipynb`.
- **Run Test**: `python src/extraction.py` processes samples and compares to ground truth.

For summarization prompts, see the notebook.

## Next Steps
- Design LLM prompts for extracting and classifying adverse events from unstructured clinical notes. (Completed: See extraction.py and notebook.)
- Integrate Gemini API (e.g., via Google AI SDK) for real-time inference on incoming EHR data. (Partially completed: Prototype in extraction.py.)
- Fuse extracted unstructured data with structured EHR for ML-based risk scoring.
- Add explainability (e.g., highlight prompt reasoning) and monitoring for API calls.
- Deploy as FastAPI endpoint with CI/CD.

## Installation and Setup
1. Clone the repo: `git clone https://github.com/was-abi/Adverse-Event-Detection-LLM.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set GEMINI_API_KEY in `.env` (get from Google AI Studio).
4. Generate data: `python ehr_data_generator.py`
5. Test extraction: `python src/extraction.py`
6. Run notebooks: `jupyter notebook notebooks/`

## To-Do Next
- [x] Set up environment and requirements.txt
- [x] Simulate/gather sample EHR data
- [x] Quick EDA notebook (notebooks/01_eda.ipynb)
- [x] Design LLM prompt/prototype Gemini API call
- [ ] Train baseline tabular ML model
- [ ] Integrate extraction into full pipeline
- [ ] Evaluate on larger dataset (e.g., full CSV)
- [ ] Deploy API with FastAPI/uvicorn
