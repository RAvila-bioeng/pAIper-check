# pAIper check
**A modular AI-based system for the ethical and scientific evaluation of academic papers**

A tool designed for the multifaceted and automated analysis of the quality, structure, and rigor of academic works. Built with students and new researchers in mind, so they can have a trustworthy and efficient tool to check the quality of papers they write or come across. Developed specifically for the **SAM Congress at the Universidad Francisco de Vitoria**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)]()

## Context: UFV SAM Congress

This project is a tool deddicated for the **IV San Alberto Magno Congress**, hosted by the Faculty of Experimental Sciences at UFV. This flagship academic event gathers students from diverse degrees, masters, and doctorate programs to share work, attend professional talks, and network.

## System Proposal

**pAIper check** implements a set of interconnected modules to provide a comprehensive and objective evaluation. Its modular design facilitates expansion and adaptation to different academic standards.

---
## Evaluation Modules
The system is organized into six fundamental pillars, each focused on a critical aspect of the quality of a scientific paper:

### 1. Structure and Completeness Assessment
This module ensures the manuscript complies with essential formal requirements.
* **Section Verification:** Checks for the presence of mandatory sections (**Abstract, Methodology, Results, References**).
* **Format Analysis:** Review of structure and formatting according to academic standards.
* **Missing Element Detection:** Identification of absent or poorly organized components.

### 2. Linguistic Quality Control
Focuses on the precision and style of scientific language.
* **Specialized Correction:** Orthographic and grammatical review tailored for scientific terminology.
* **Terminology Consistency:** Verification that technical terms are used uniformly throughout the text.
* **Appropriate Academic Style:** Analysis of the tone and formality required for scientific communication.

### 3. Coherence and Cohesion Analysis
Evaluates the flow and logical connection of the argument.
* **Argumentative Fluency:** Assessment of how ideas are developed throughout the manuscript.
* **Connectivity:** Analysis of transitions and the logical relationship between paragraphs and sections.
* **Narrative Consistency:** Ensures a uniform storyline from the introduction to the conclusions.

### 4. Reproducibility Assessment
A crucial pillar for modern research, centered on methodological transparency.
* **Methodological Clarity:** Analysis of the sufficiency of method descriptions to allow replication.
* **Data Availability:** Verification of mentions regarding accessibility to data and/or code.
* **Replicability:** Evaluation of the theoretical possibility of reproducing the experiments or analysis.

### 5. References and Citation Verification
Guarantees the work's strength and bibliographical support.
* **Automated Validation:** Checks the correct notation and format of bibliographic citations.
* **Existence and Access:** Verification of the accessibility and validity of the cited sources.
* **Credibility and Relevance:** Analysis of the currency and pertinence of the references used.

### 6. Scientific Quality Evaluation
The core of the review, focusing on contribution and intrinsic rigor.
* **Novelty and Originality:** Analysis of the paper's contribution to the existing body of knowledge.
* **Methodological Rigor:** Evaluation of the suitability of the selected methods for the stated objectives.
* **Significance of Results:** Validation of the importance and implications of the findings.

---
## System Architecture
The system is designed with a modular approach, facilitating maintenance and the integration of new evaluation criteria.

1.  **Input:** The manuscript is provided (preferably in **PDF** or plain text format).
2.  **Preprocessing:** Extraction and segmentation of the document's key sections (Title, Abstract, Conclusions, etc.).
3.  **Modular Analysis:** Each text segment is passed to the corresponding evaluation modules (Structure, Linguistic, References, etc.).
4.  **Consolidation and Output:** Results from each module are compiled and processed to generate a **Detailed Evaluation Report**, including scores by category and constructive feedback.
