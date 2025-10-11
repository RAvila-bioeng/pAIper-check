# pAIper-check
AI-powered auditing tool for scientific papers, focused on rigor, reproducibility, and hype detection.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue)]()

In the era of rapid growth in scientific literature, verifying the **quality**, **methodological rigor**, and **veracity** of papers has become a bottleneck for editors and reviewers.

**pAIper Check** uses **Large Language Models (LLMs)** and scripting to perform a multidimensional evaluation of manuscripts, assisting reviewers and providing researchers with a tool for critical pre-review.


## Key features:

pAIper Check offers two main usage modes ('settings')—Researcher and Reviewer—and focuses on the following pillars:

## 1. Scientific Rigor and Ethics (Veracity)
Hype Filter: Detects exaggerated language and bias in titles and conclusions, cross-checking them against the robustness of the results.

Ethics Verification: Searches for ethics committee approval and the use of informed consent.

Premise Identification: Ensures direct alignment between the initial hypothesis and the conclusions.

Critical Citations: Verifies that key claims are supported by current or foundational references.

## 2. Cohesion and Reproducibility
Reproducibility Score: Evaluates methodological clarity and the availability of datasets, code, or materials for replicability.

Figure-Text Alignment: Validates visual references and Figure Caption Quality (must be self-explanatory).

Linguistic Control: Checks consistency of technical terminology and argumentative fluency.

## 3. Advanced Output and Feedback (AI-Powered)
Dynamic Spider Chart: Provides a clear, immediate visualization of quality scores by category.

Author Recommendations: Generates constructive, specific feedback to improve the manuscript.

LLM-Generated Text Detection (Experimental): Module that identifies text patterns suggesting automatic generation.

## System architecture:
The **pAIper Check** workflow is modular and centers on LLM abstraction for easy integration.

Input: The user provides a PDF file (or text) and the desired setting ('--reviewer' or '--researcher').

Preprocessing: Key sections are extracted (Title, Abstract, Conclusion, Methods, etc.).

Modular Analysis: Each section is passed to the corresponding analysis modules (Hype, Ethics, Reproducibility), which interact with the LLM through specific prompts.

Scoring & Output: The results from each module are consolidated to generate the Spider Chart and final Recommendations.
