# Resume Classification and Ranking System

## Overview
This project builds an AI pipeline to:
1. Classify resumes into categories
2. Rank resumes based on relevance to a job description

## Project Structure
- `src/resume_pipeline/`: Resume classification modules
- `src/ranking_pipeline/`: Ranking modules
- `src/utils/`: Shared utilities
- `models/`: Saved ML artifacts
- `data/`: Raw and processed data
- `notebooks/`: Experiments and analysis

## Team Roles
- **Fares**: Resume classification pipeline
- **Mohab**: Ranking pipeline
- **Shared**: Integration, utilities, documentation

## Planned Workflow
1. Preprocess resumes
2. Extract features
3. Train classification model
4. Process job description
5. Compute similarity and rank resumes
6. Integrate into one pipeline

## Setup
```bash
pip install -r requirements.txt