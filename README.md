# pAIper check
**A modular AI-based system for the ethical and scientific evaluation of academic papers**

A tool designed for the multifaceted and automated analysis of the quality, structure, and rigor of academic works. Built with students and new researchers in mind, so they can have a trustworthy and efficient tool to check the quality of papers they write or come across. Developed specifically for the **SAM Congress at the Universidad Francisco de Vitoria**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue)]()

## Context: UFV SAM Congress

This project is a tool deddicated for the **IV San Alberto Magno Congress**, hosted by the Faculty of Experimental Sciences at UFV. This flagship academic event gathers students from diverse degrees, masters, and doctorate programs to share work, attend professional talks, and network.

## System Proposal

**pAIper check** implements a set of interconnected modules to provide a comprehensive and objective evaluation. Its modular design facilitates expansion and adaptation to different academic standards.

---
## Evaluation Modules
The system is organized into six fundamental pillars, each focused on a critical aspect of the quality of a scientific paper. The base analysis relies on a combination of heuristic-based checks and natural language processing, while the optional LLM-powered analysis provides a deeper, more nuanced evaluation.

### 1. Structure and Completeness Assessment
This module ensures the manuscript complies with standard academic structure.
* **Section Verification:** Checks for the presence of **Abstract, Introduction, Methods, Results, Discussion, Conclusion, and References**, accepting common synonyms.
* **Content Analysis:** Assesses if each section has a reasonable length.
* **Order Verification:** Ensures sections follow the logical IMRaD (Introduction, Methods, Results, and Discussion) sequence.
* **Title Quality:** Evaluates the title for appropriate length and capitalization.

### 2. Linguistic Quality Control
Focuses on the precision, clarity, and style of scientific language.
* **Spelling and Grammar:** Performs a spelling check using a custom dictionary of academic terms to reduce false positives. It also identifies common grammatical and punctuation errors.
* **Academic Style:** Detects and flags informal language (e.g., "lots of") and contractions (e.g., "don't") that are inappropriate for formal writing.
* **Terminology Consistency:** Checks for consistent use of technical terms (e.g., flags mixing "dataset" and "data set").
* **Readability:** Calculates readability scores (Flesch Reading Ease) and interprets them within an academic context.

### 3. Coherence and Cohesion Analysis
Evaluates the logical flow and connectivity of the paper's argument through four weighted sub-metrics:
* **Argumentative Fluency:** Measures the density of logical connectors (e.g., "however," "therefore") to assess the strength of the argument.
* **Section Connectivity:** Counts explicit cross-references (e.g., "as shown in Section 3") to evaluate how well different parts of the paper are linked.
* **Narrative Consistency:** Analyzes the consistent use of key technical terms and point of view (e.g., first vs. third person).
* **Logical Flow:** Verifies that the sections follow the standard IMRaD order.

### 4. Reproducibility Assessment
A crucial pillar for modern research, centered on methodological transparency. The evaluation is based on detecting specific, replicable details.
* **Methodological Clarity:** Scans the text for precise, quantitative descriptions (e.g., measurements with units) and penalizes vague language (e.g., "approximately" without a number).
* **Data and Code Availability:** Checks for statements indicating where data or code can be found (e.g., mentions of GitHub, Zenodo, or supplementary materials).
* **Materials Specification:** Looks for detailed descriptions of materials and equipment, including vendor names and catalog numbers.
* **Parameter Specification:** Verifies that key experimental parameters (e.g., temperature, pH, concentration) are clearly defined.

### 5. References and Citation Verification
Guarantees the work's bibliographical support through heuristic-based analysis.
* **Format Consistency:** Infers the citation style (e.g., numbered or author-year) and checks for consistency across all references.
* **Quantity Analysis:** Evaluates if the number of references is appropriate for a scientific paper.
* **Recency Analysis:** Calculates the percentage of references from the last 10 years to assess the currency of the sources.
* **Source Diversity:** Checks for a healthy mix of source types (e.g., journals, books, web).
* **LLM Deep Analysis (Optional):** When enabled, uses the Perplexity API to perform a more advanced analysis of the references' relevance and quality.

### 6. Scientific Quality Evaluation
This module provides a heuristic-based "first look" at the paper's potential contribution and rigor. A deep, meaningful evaluation is performed when the optional LLM analysis is enabled.
* **Novelty and Originality:** Scans for keywords indicating novelty (e.g., "novel," "first time," "propose") and explicit statements of contribution.
* **Methodological Rigor:** Looks for indicators of a robust methodology, such as mentions of objectives, hypotheses, and statistical methods.
* **Significance of Results:** Detects quantitative results, comparisons to baselines, and statements of statistical significance (e.g., p-values).
* **LLM Deep Analysis (Optional):** When enabled, uses the OpenAI API (GPT-4o-mini) to perform a holistic evaluation of the paper's originality, rigor, and the significance of its findings.

---
## System Architecture
The system is designed with a modular approach, facilitating maintenance and the integration of new evaluation criteria.

1.  **Input:** The manuscript is provided (preferably in **PDF** or plain text format).
2.  **Preprocessing:** Extraction and segmentation of the document's key sections (Title, Abstract, Conclusions, etc.).
3.  **Modular Analysis:** Each text segment is passed to the corresponding evaluation modules (Structure, Linguistic, References, etc.).
4.  **Consolidation and Output:** Results from each module are compiled and processed to generate a **Detailed Evaluation Report**, including scores by category and constructive feedback.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/paiper-check.git
    cd paiper-check
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the necessary spaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Configuration

For deep analysis features using Large Language Models (LLMs), you need to configure API keys.

1.  Create a `.env` file in the root of the project.
2.  Add your API keys to the file:
    ```
    OPENAI_API_KEY="sk-..."
    PERPLEXITY_API_KEY="..."
    ```
    *   `OPENAI_API_KEY`: Used for the deep analysis in the **Scientific Quality** and **Cohesion** modules.
    *   `PERPLEXITY_API_KEY`: Used for the deep analysis in the **References** module.

    *Note: The application will run without these keys, but the `--use-llm` flag will be disabled.*

## Usage

The application is run from the command line. Here are the most common commands:

### Basic Evaluation
To run a standard analysis on a single paper:
```bash
python src/main.py --input path/to/your/paper.pdf
```

### Deep Analysis with LLMs
To enable the deeper, more accurate analysis using OpenAI and Perplexity models, use the `--use-llm` flag:
```bash
python src/main.py --input path/to/your/paper.pdf --use-llm
```

### Displaying Linguistic Errors
To see a detailed list of spelling and grammar mistakes, use `--show-errors`:
```bash
python src/main.py --input path/to/your/paper.pdf --show-errors
```

### Saving the Report
To save the full analysis report to a JSON file, use the `--output` flag:
```bash
python src/main.py --input paper.pdf --use-llm --output results.json
```

### All Options
```
usage: main.py [-h] --input INPUT [--output OUTPUT] [-v] [--use-llm]
               [--force-gpt] [--gpt-report] [--batch] [--show-errors]
               [--max-errors MAX_ERRORS] [--export-errors EXPORT_ERRORS]

Run pAIper Check evaluation with optional GPT-4o-mini deep analysis

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to input paper (PDF or TXT)
  --output OUTPUT       Path to save evaluation report (JSON)
  -v, --verbose         Verbose output
  --use-llm             Enable deep analysis with GPT-4o-mini and Perplexity Sonar
  --force-gpt           Force GPT analysis even if basic score is good
  --gpt-report          Show detailed GPT cost report at the end
  --batch               Process multiple papers (use with wildcards)
  --show-errors         Show detailed linguistic errors with locations
  --max-errors MAX_ERRORS
                        Maximum number of errors to display (default: 30)
  --export-errors EXPORT_ERRORS
                        Export linguistic errors to file (JSON, CSV, or HTML)
```
