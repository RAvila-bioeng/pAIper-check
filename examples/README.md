# pAIper Check - Examples

This directory contains examples of how to use pAIper Check. The primary way to use this tool is through the command-line interface (CLI) via `src/main.py`.

## Basic Usage

To run a basic analysis on a scientific paper, use the following command:

```bash
python src/main.py --input path/to/your/paper.pdf
```

This will run all the standard analysis modules (structure, linguistics, cohesion, etc.) without using any external LLM APIs. This analysis is fast and completely free.

## Deep Analysis with LLMs

To get a more in-depth analysis, you can enable the `--use-chatgpt` flag. This will use LLMs (like GPT-4o-mini and Perplexity Sonar Pro) to enhance the evaluation of each pillar.

**Note:** This requires you to have API keys set up in a `.env` file. See the main `README.md` for instructions.

```bash
python src/main.py --input path/to/your/paper.pdf --use-chatgpt
```

## Saving the Report

You can save the full analysis report to a JSON file using the `--output` flag. This is useful for programmatic access to the results.

```bash
python src/main.py --input path/to/your/paper.pdf --use-chatgpt --output results.json
```

## Viewing Linguistic Errors

If you want to see a detailed breakdown of spelling, grammar, and style errors, use the `--show-errors` flag.

```bash
python src/main.py --input path/to/your/paper.pdf --show-errors
```

You can control the number of errors displayed with `--max-errors`:

```bash
python src/main.py --input path/to/your/paper.pdf --show-errors --max-errors 50
```

## Batch Processing

If you have multiple papers to analyze, you can use the `--batch` flag with a wildcard in the input path.

```bash
# Analyze all PDF files in the 'papers' directory
python src/main.py --input "papers/*.pdf" --use-chatgpt --batch --output batch_results.json
```
