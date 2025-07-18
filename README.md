# AI-Powered Call Transcript Processor
This script uses the google/flan-t5-large model to automatically extract structured survey answers from raw call transcripts. It fetches unprocessed call logs from a Supabase database, uses the AI to find answers to predefined questions, and saves the structured results back to Supabase.

This project is designed to run in a Google Colab environment to leverage GPU acceleration and secure secret management.

## Core Functionality
Connects to a Supabase database and loads the FLAN-T5 model.
Fetches call logs that have not yet been processed (is_classified is false).
Iterates through each transcript, using the AI to find answers to a list of questions.
Saves the extracted answers to a survey_results table.
Updates the original call log (is_classified to true) to prevent reprocessing.

## Usage
Add the Python script to a cell in your Google Colab notebook.

Populate your call_logs table with transcripts.

Run the cell. The script will process all unclassified calls and save the results to the survey_results table.

## Python Requirements
```bash
pip install --upgrade torch torchvision torchaudio supabase transformers sentencepiece accelerate bitsandbytes -q
```
## Note :
use it in google cloab
