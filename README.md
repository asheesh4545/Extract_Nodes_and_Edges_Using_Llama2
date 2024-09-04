# Extract_Nodes_and_Edges_Using_Llama2

# Knowledge Graph Extraction from Medical Texts

This repository contains two Python scripts for extracting knowledge graphs from medical texts using Large Language Models (LLMs) via the Groq API. Both scripts aim to create structured representations of medical information but differ in their approach to knowledge application.

## Overview

1. `medical_kg_extractor.py`: Extracts information strictly based on the given ontology without using pre-existing knowledge.
2. `product_kg_extractor.py`: Utilizes both the given ontology and the model's pre-existing knowledge for information extraction.

## Key Differences

- **Knowledge Application:**
  - `medical_kg_extractor.py`: Adheres strictly to the provided ontology without incorporating external knowledge.
  - `product_kg_extractor.py`: Combines the provided ontology with the model's pre-existing knowledge for more flexible extraction.

- **Ontology Structure:**
  - `medical_kg_extractor.py`: Uses a more detailed ontology with specific entity types and relationships.
  - `product_kg_extractor.py`: Employs a broader ontology allowing for more general entity and relationship types.

- **Validation Step:**
  - `medical_kg_extractor.py`: Includes an additional validation step to ensure logical consistency of extracted relationships.
  - `product_kg_extractor.py`: Relies on a single-pass extraction without explicit validation.

## Requirements

- Python 3.6+
- `groq` Python client
- `python-dotenv`

## Installation

1. Clone this repository:
git clone https://github.com/asheesh4545/Extract_Nodes_and_Edges_Using_Llama2.git
cd Extract_Nodes_and_Edges_Using_Llama2

2. Install the required packages:
pip install groq python-dotenv

3. Set up your Groq API key:
- Create a `.env` file in the project root
- Add your Groq API key: `GROQ_API_KEY=your_api_key_here`

## Usage

### Medical KG Extractor (Strict Ontology Adherence)

```python
python medical_kg_extractor.py



1. Clone this repository:
