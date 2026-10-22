# SelfCheckGPT
### Hallucination Detection and Visualization in Large Language Model Outputs

**A Zero-Resource, Self-Consistency-Based Framework for Explainable AI Reliability Assessment**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-000000?logo=flask&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini_API-4285F4?logo=google&logoColor=white)
![NLP](https://img.shields.io/badge/Domain-Natural_Language_Processing-8A2BE2)
![BERTScore](https://img.shields.io/badge/Evaluation-BERTScore-00897B)
![Research](https://img.shields.io/badge/Category-AI_Research-6C63FF)

---

## Overview

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language generation, reasoning, summarization, and knowledge-intensive question answering. However, their widespread adoption introduces a fundamental reliability challenge: **hallucination**.

An LLM may generate statements that are fluent, coherent, and linguistically convincing while containing factually inaccurate, unsupported, or entirely fabricated information.

**SelfCheckGPT** is a research-oriented hallucination detection and visualization system designed to analyze the factual consistency of LLM-generated responses without requiring external knowledge bases, manually constructed fact-checking rules, or retrieval-based verification.

The system combines **Google Gemini, stochastic response sampling, SelfCheckGPT-inspired consistency analysis, and contextual semantic similarity using BERTScore** to estimate potential hallucinations at the sentence level.

It transforms these estimates into an interpretable, color-coded interface, allowing users to inspect individual sentences, identify potentially unreliable claims, and understand the overall consistency of generated content.

Rather than treating an LLM response as a single trustworthy or untrustworthy unit, SelfCheckGPT evaluates its constituent statements individually.

### Core Principle

> A statement that remains semantically consistent across independently sampled responses is treated as more internally supported, while a statement that varies significantly across samples is considered potentially unreliable.

This is a probabilistic consistency heuristic, not definitive proof of factual correctness.

---

## 1. Problem Statement

Modern LLMs generate responses through probabilistic language modeling rather than guaranteed factual verification.

This creates several challenges:

- **Factual hallucinations:** Generation of incorrect dates, locations, names, statistics, or historical claims.
- **Fabricated information:** Production of plausible but nonexistent events, references, or entities.
- **Overconfident generation:** Presentation of uncertain information using authoritative language.
- **Limited interpretability:** Difficulty identifying which specific portions of a response may be unreliable.
- **Verification overhead:** Dependence on manual fact-checking or external information sources.

Traditional fact-checking pipelines often require reference documents, knowledge graphs, retrieval systems, or supervised verification datasets.

SelfCheckGPT investigates an alternative approach: estimating factual consistency by comparing multiple responses generated from the same underlying model.

---

## 2. Research Objectives

The primary objective is to develop an interpretable framework for detecting and visualizing potential hallucinations in LLM-generated text.

Specific objectives include:

1. Generate a primary response and multiple stochastic samples using an LLM.
2. Measure sentence-level semantic consistency across independently generated responses.
3. Estimate hallucination likelihood using a BERTScore-based self-consistency formulation.
4. Classify sentences into factual, uncertain, and hallucinated categories using configurable thresholds.
5. Calculate an aggregate passage-level factuality estimate.
6. Present results through an interactive, color-coded web dashboard.
7. Maintain user-specific analysis history for subsequent inspection.
8. Evaluate detection performance using a publicly available, human-annotated hallucination dataset.
9. Analyze metric reliability using bootstrap resampling and correlation-based evaluation.

---

## 3. System Architecture

The framework follows a modular architecture separating response generation, semantic analysis, score aggregation, visualization, and persistent storage.

```text
                    USER INPUT
                        |
                        v
                FLASK WEB INTERFACE
                        |
                        v
                  GEMINI API
                        |
             +----------+----------+
             |                     |
             v                     v
       PRIMARY RESPONSE      STOCHASTIC SAMPLES
        Temperature = 0       Temperature = 1
             |                     |
             +----------+----------+
                        |
                        v
                TEXT SEGMENTATION
                        |
                        v
              SELF-CONSISTENCY ENGINE
                        |
                        v
                BERTScore ANALYSIS
                        |
                        v
              SENTENCE-LEVEL SCORING
                        |
                        v
                THRESHOLD MAPPING
                        |
             +----------+----------+
             |          |          |
           FACTUAL   UNCERTAIN  HALLUCINATED
             |          |          |
             +----------+----------+
                        |
                        v
               PASSAGE-LEVEL SCORE
                        |
                        v
               VISUALIZATION ENGINE
                        |
             +----------+----------+
             |                     |
             v                     v
       ANALYSIS DASHBOARD      SQLITE HISTORY
```

### Architectural Characteristics

**Modularity:** Individual components are logically separated, supporting independent modification and testing.

**Interpretability:** Sentence-level scores are exposed instead of presenting only a single opaque prediction.

**Reference-free inference:** The core consistency analysis does not require externally retrieved evidence.

**Persistence:** Analysis records can be stored for authenticated users.

**Extensibility:** The architecture can accommodate alternative semantic similarity models, additional evaluation metrics, and future verification modules.

---

## 4. Methodology

### 4.1 Primary Response Generation

The system receives a natural-language prompt and submits it to the Gemini API.

A primary response is generated using a low-temperature configuration.

In the experimental configuration:

```text
Primary Response Temperature: 0
```

This promotes comparatively deterministic response generation.

The primary response is segmented into individual sentences for subsequent analysis.

### 4.2 Stochastic Response Sampling

To estimate self-consistency, the system generates additional responses to the same prompt using a higher sampling temperature.

```text
Stochastic Sampling Temperature: 1
```

Higher-temperature sampling encourages variation in generated wording and content.

The generated responses serve as alternative semantic realizations against which the primary response is evaluated.

The framework has been investigated using different sample counts, including:

- N = 1
- N = 3

Increasing the number of samples may improve the stability of the consistency estimate but introduces additional API calls and computational cost.

### 4.3 Semantic Similarity Using BERTScore

Lexical comparison methods can incorrectly classify semantically equivalent sentences as inconsistent when they use different vocabulary.

To address this, SelfCheckGPT employs **BERTScore**, a contextual semantic similarity metric based on pretrained transformer representations.

BERTScore compares contextualized token embeddings rather than relying exclusively on exact word overlap.

For two text sequences, contextual token representations are compared using cosine similarity.

The resulting similarities are aggregated into precision, recall, and F1-based measures.

The implementation uses a RoBERTa-Large-based contextual similarity configuration.

### 4.4 Sentence-Level Hallucination Estimation

Let:

- \(r_i\) represent the \(i\)-th sentence in the primary response.
- \(S^n\) represent the \(n\)-th stochastic response.
- \(s_k^n\) represent the \(k\)-th sentence in stochastic response \(n\).
- \(N\) represent the number of stochastic responses.

The sentence-level hallucination score is estimated as:

\[
H(r_i)=1-\frac{1}{N}\sum_{n=1}^{N}\max_k \operatorname{BERTScore}_{F1}(r_i,s_k^n)
\]

For each primary sentence, the system identifies its most semantically similar sentence in every sampled response.

The maximum similarity values are averaged across samples, and the result is inverted to obtain an inconsistency estimate.

**Interpretation:**

- Lower inconsistency indicates stronger agreement across sampled responses.
- Higher inconsistency indicates greater semantic variation and potentially unreliable content.

A high inconsistency score is a hallucination warning, not an independently verified factual error.

### 4.5 Threshold-Based Classification

The system maps sentence-level inconsistency estimates into three interpretable categories.

| Hallucination Score | Classification | Interpretation |
|---|---|---|
| 0.00–0.33 | Factual | Stronger semantic consistency |
| 0.34–0.66 | Uncertain | Moderate semantic inconsistency |
| 0.67–1.00 | Hallucinated | Substantial semantic inconsistency |

These are application-defined heuristic thresholds rather than universally calibrated probabilities of factual correctness.

### 4.6 Passage-Level Factuality Estimation

Sentence-level estimates are aggregated to produce a passage-level score:

\[
F_p=1-\frac{1}{M}\sum_{i=1}^{M}H(r_i)
\]

Where \(M\) is the number of sentences in the primary response.

Higher values indicate greater average semantic consistency across the analyzed passage.

The passage-level score provides a compact summary while preserving detailed sentence-level results for interpretation.

---

## 5. Key Features

### 5.1 LLM Response Generation

The system integrates the Google Gemini API to generate natural-language responses from user prompts.

Primary and stochastic responses support the self-consistency evaluation workflow.

### 5.2 Sentence-Level Hallucination Detection

Each sentence is analyzed independently using contextual semantic similarity.

This enables the identification of potentially inconsistent statements without assigning a uniform score to an entire response.

### 5.3 Explainable Color-Coded Visualization

Analyzed sentences are presented using three visual indicators:

🟢 **Green — Factual:** Relatively strong agreement across sampled responses.

🟡 **Yellow — Uncertain:** Partial semantic agreement or noticeable variation.

🔴 **Red — Hallucinated:** Significant inconsistency requiring additional verification.

The interface is designed to make model reliability signals understandable without requiring users to interpret raw embedding representations.

### 5.4 Sentence-Level Score Inspection

Users can inspect the numerical hallucination estimate associated with individual sentences.

This provides greater transparency than displaying only a categorical label.

### 5.5 Aggregate Factuality Dashboard

The dashboard summarizes the analyzed response through an overall factuality estimate.

It complements sentence-level inspection by providing a high-level consistency indicator.

### 5.6 User Authentication and Analysis History

The application includes user registration, login, session handling, and persistent analysis history.

SQLite provides lightweight relational storage for local application data.

The authentication workflow restricts analysis access according to the application's login requirements.

### 5.7 Research Evaluation Pipeline

The project includes an evaluation component designed to compare model-derived hallucination scores with human-annotated factuality judgments.

The evaluation workflow supports ranking-based metrics, passage-level correlations, and bootstrap-based uncertainty analysis.

---

## 6. Evaluation and Experimental Design

### 6.1 Benchmark Dataset

The evaluation framework uses the **SelfCheckGPT WikiBio hallucination dataset**, which contains LLM-generated biographical passages and human-annotated factuality information.

The experimental setup considers:

| Property | Description |
|---|---|
| Dataset | SelfCheckGPT WikiBio |
| Passages | 238 |
| Reference sampling structure | 20 sampled responses per passage |
| Annotation granularity | Sentence level |
| Labels | Accurate, Minor Inaccurate, Major Inaccurate |
| Evaluation approach | Ranking and correlation analysis |
| Bootstrap iterations | 1,000 |

The benchmark dataset is distinct from the live Gemini generation workflow. Its passages and samples are used for offline evaluation, not automatically retrieved as references for each new user prompt.

### 6.2 Evaluation Tasks

The evaluation methodology considers three factuality-oriented tasks.

**NonFact:** Evaluate the ability to rank nonfactual statements above accurate statements.

**NonFact\*:** Evaluate a stricter nonfactuality definition, emphasizing major factual inaccuracies.

**Factual:** Evaluate the identification of factually accurate statements.

The exact binary label mappings must remain consistent with the benchmark evaluation protocol.

### 6.3 Evaluation Metrics

**Area Under the Precision–Recall Curve (AUC-PR)**

Measures ranking quality across different decision thresholds, with particular relevance when class distributions are imbalanced.

**Pearson Correlation**

Measures the strength of the linear relationship between predicted passage scores and reference factuality measurements.

**Spearman Rank Correlation**

Measures how consistently predicted scores preserve the ranking of passages according to their reference factuality.

### 6.4 Bootstrap Resampling

Bootstrap resampling is used to estimate uncertainty in evaluation metrics.

The procedure consists of:

1. Sampling benchmark examples with replacement.
2. Constructing a bootstrap dataset.
3. Recalculating the selected evaluation metric.
4. Repeating the procedure 1,000 times.
5. Analyzing the resulting metric distribution.

This supports the estimation of variability and confidence intervals when the evaluation pipeline computes and reports them.

**Reporting note:** Numerical performance claims should be accompanied by reproducible experimental results. This repository description intentionally does not present unverified AUC-PR, correlation, or confidence-interval values.

---

## 7. Technology Stack

| Layer | Technologies |
|---|---|
| Programming Language | Python |
| Backend Framework | Flask |
| LLM Integration | Google Gemini API |
| Hallucination Analysis | SelfCheckGPT-inspired consistency scoring |
| Semantic Similarity | BERTScore |
| Transformer Backbone | RoBERTa-Large |
| NLP Processing | spaCy |
| Deep Learning | PyTorch, Hugging Face Transformers |
| Frontend | HTML, CSS, JavaScript |
| Template Engine | Jinja2 |
| Database | SQLite |
| Configuration | Environment Variables, python-dotenv |
| Evaluation | Hugging Face Datasets, statistical evaluation tools |
| Development Environment | Visual Studio Code |

---

## 8. Repository Structure

The following structure illustrates the principal application components.

```text
Hallucination-Detection-System/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── services/
│   ├── llm_service.py
│   └── database_service.py
│
├── evaluation/
│   └── ... evaluation scripts and resources
│
├── static/
│   └── ... CSS and JavaScript assets
│
└── templates/
    ├── index.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    └── history.html
```

**Component Responsibilities**

`app.py` — Flask application entry point, routing, and request handling.

`services/` — LLM integration, hallucination analysis services, and database operations.

`evaluation/` — Benchmark experiments, score evaluation, and statistical analysis.

`templates/` — HTML templates for authentication, analysis, dashboards, and history.

`static/` — Frontend stylesheets and JavaScript resources.

`requirements.txt` — Python package dependencies.

---

## 9. Installation and Setup

### Prerequisites

- Python and pip
- Git
- A valid Google Gemini API key
- Internet access for API requests and initial model downloads
- Sufficient system resources for transformer-based semantic evaluation

### Clone the Repository

```bash
git clone https://github.com/Saloni15apr/Hallucination-Detection-System.git
cd Hallucination-Detection-System
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Activate on Linux or macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

If the Gemini SDK is missing:

```bash
python -m pip install google-genai
```

### Configure Environment Variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key
```

The environment variable name must match the configuration expected by the application.

**Security:** Never commit API keys, local credentials, user databases, or virtual environments to version control.

### Run the Application

```bash
python app.py
```

For the default Flask development configuration, open:

```text
http://127.0.0.1:5000
```

The development server is intended for local testing, not production deployment.

---

## 10. Example Analysis Workflow

Consider the illustrative input:

```text
VIT is located in Ahmedabad.
```

The framework follows this process:

**Stage 1 — Primary Generation**

Gemini generates an initial response to the submitted prompt.

**Stage 2 — Alternative Sampling**

Additional responses are generated using stochastic sampling.

**Stage 3 — Sentence Segmentation**

The primary and sampled responses are divided into sentence-level units.

**Stage 4 — Semantic Comparison**

BERTScore compares each primary sentence with the most semantically similar sentences in the sampled responses.

**Stage 5 — Inconsistency Estimation**

The framework computes a sentence-level hallucination estimate.

**Stage 6 — Visualization**

The resulting estimate is mapped to a color-coded classification and displayed in the dashboard.

**Important distinction:** The system does not independently establish the correct location of an institution from an authoritative database. It estimates whether the model's generated claims are consistent across samples.

---

## 11. Research Significance

SelfCheckGPT explores an important question in trustworthy language-model development:

**Can a language model's own generation variability provide useful evidence about the reliability of its responses?**

The project contributes an integrated implementation combining:

- Reference-free consistency estimation.
- Contextual semantic comparison.
- Sentence-level uncertainty visualization.
- Passage-level reliability summaries.
- Authenticated analysis history.
- Benchmark-oriented evaluation workflows.

Its engineering contribution lies in integrating these components into an accessible application and exposing sentence-level consistency signals to users.

The underlying SelfCheckGPT and BERTScore methodologies are established research techniques; this implementation does not claim to have invented those methods.

---

## 12. Limitations

Despite its practical utility, self-consistency-based hallucination detection has important limitations.

### Consistent Hallucinations

An LLM may repeatedly generate the same incorrect statement. High agreement across samples does not guarantee factual correctness.

### Sampling Sensitivity

Results may vary depending on sampling temperature, sample count, model version, and prompt formulation.

### Semantic Similarity Limitations

High semantic similarity does not necessarily imply factual equivalence, particularly for numerical claims, dates, named entities, and subtle contradictions.

### Computational Overhead

Generating multiple responses and performing transformer-based semantic comparisons increases latency and resource consumption.

### Threshold Calibration

The classification thresholds are heuristic and may not generalize equally across different domains, languages, and model families.

### External Verification

The framework does not replace authoritative fact-checking, source attribution, or human review in high-stakes applications.

---

## 13. Future Research Directions

The framework can be extended in several directions.

**Hybrid Retrieval-Augmented Verification**

Combine self-consistency estimates with evidence retrieved from reliable external sources.

**Natural Language Inference**

Introduce entailment and contradiction detection to distinguish semantic similarity from factual agreement.

**Adaptive Sampling**

Dynamically determine the number of stochastic responses required based on score stability.

**Confidence Calibration**

Investigate data-driven threshold calibration and evaluate whether predicted inconsistency scores correspond to empirical factual-error rates.

**Multilingual Evaluation**

Extend evaluation beyond English-language text to study consistency across languages.

**Entity-Level Verification**

Introduce targeted verification of names, dates, locations, quantities, and other factual entities.

**Model Comparison**

Benchmark hallucination consistency patterns across multiple language-model families.

**Improved Explainability**

Visualize sample-to-sentence matches and provide evidence for individual inconsistency estimates.

These are proposed research extensions and should not be interpreted as currently implemented features.

---

## 14. Applications

Potential applications include:

- LLM response quality assessment.
- AI-assisted content review.
- Research on hallucination detection.
- Educational demonstrations of trustworthy AI.
- Semantic consistency benchmarking.
- Exploratory reliability analysis of generated text.
- Human-in-the-loop verification workflows.

The system is intended to support informed review rather than replace independent factual verification.

---

## 15. Research Foundations

This project builds on established work in hallucination detection and contextual semantic evaluation.

**SelfCheckGPT**

Manakul, P., Liusie, A., & Gales, M. J. F. (2023).

*SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.*

Proceedings of EMNLP 2023.

https://arxiv.org/abs/2303.08896

**BERTScore**

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020).

*BERTScore: Evaluating Text Generation with BERT.*

International Conference on Learning Representations (ICLR).

https://arxiv.org/abs/1904.09675

These references acknowledge the original methodologies on which the implementation is based.

---

## 16. Responsible Use

SelfCheckGPT provides model-consistency estimates and visual indicators of potentially unreliable generated statements.

The terms *factual*, *uncertain*, and *hallucinated* refer to application-level classification categories based on consistency scores.

They should not be interpreted as externally verified truth labels.

Outputs should be independently verified before use in medical, legal, financial, scientific, or other consequential decision-making contexts.

---

## 17. Project Information

**Project:** SelfCheckGPT — Hallucination Detection and Visualization System

**Domain:** Artificial Intelligence, Natural Language Processing, Explainable AI, Trustworthy Machine Learning

**Project Type:** Research-Oriented Software Implementation

**Developer:** Saloni Ghutukade

**GitHub:** https://github.com/Saloni15apr/Hallucination-Detection-System

**Development Status:** Functional research prototype; continued evaluation and refinement.

---

### Closing Statement

**Reliable AI requires more than fluent generation. It requires mechanisms to examine uncertainty, expose potential inconsistencies, and support transparent evaluation.**

SelfCheckGPT investigates how self-consistency sampling and contextual semantic analysis can make hallucination-related signals more visible, interpretable, and measurable.

*From generated answers to interpretable reliability signals.*
