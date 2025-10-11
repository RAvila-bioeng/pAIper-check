# pAIper Check – System Architecture

## Overview
pAIper Check is a modular evaluation system that analyzes scientific manuscripts across multiple quality dimensions.

## Flow
1. **Input:** PDF or plain text file uploaded.
2. **Preprocessing:** Extracts and segments text (Title, Abstract, Sections...).
3. **Analysis Modules:** Each module independently evaluates one of the six pillars.
4. **Aggregation:** Scores and feedback from all modules are consolidated into a `Report` object.
5. **Output:** A JSON and/or visual report summarizing the evaluation.

## Key Components
- `Paper`: Represents the manuscript.
- `PillarResult`: Stores results from one evaluation pillar.
- `Report`: Aggregates all results and computes a final score.
