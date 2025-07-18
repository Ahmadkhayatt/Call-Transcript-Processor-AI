import os
import json
from supabase import create_client, Client
from google.colab import userdata
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("⚙️ Setting up Supabase client and AI model...")

# --- Connect to Supabase using secrets ---
try:
    supabase_url = userdata.get('SUPABASE_URL')
    supabase_key = userdata.get('SUPABASE_KEY')
    supabase: Client = create_client(supabase_url, supabase_key)
    print("✅ Supabase client connected.")
except Exception as e:
    print(f"❌ ERROR: Could not connect to Supabase. Check your URL and Key in Secrets. Details: {e}")
    raise


#upgrading to the 'large' version of FLAN-T5 
model_name = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
print(f" AI Model '{model_name}' loaded onto {device}.")


def find_answer_for_question(transcript: str, question: str) -> str:
    """
    Uses the AI model to find the answer to a SINGLE question within the transcript.
    """
    prompt = f"""
You are a data extraction expert. Your task is to find the user's answer to a specific question from a conversation transcript.

INSTRUCTIONS:
1. First, locate the exact question in the transcript.
2. Then, analyze the user's response that IMMEDIATELY follows that question.
3. Extract only the core information from that specific user response.
4. If the user's response is a refusal, off-topic, or doesn't answer the question (e.g., "Kapat", "Cevap vermek istemiyorum"), you MUST respond with "Cevap Vermedi".
5. If you cannot find the exact question in the transcript at all, you MUST respond with "Soru Sorulmadı".

TRANSCRIPT:
---
{transcript}
---

QUESTION:
"{question}"

CONCISE ANSWER:
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=50) # Answers are usually short
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return answer
    except Exception as e:
        print(f"   ❌ Error during model generation: {e}")
        return "İşlem Hatası"


def process_unclassified_calls():
    print("\n🚀 Starting the classification process with the new strategy...")

    # 1. Define the questions and their corresponding database columns
    questions_to_ask = {
        "katilim_onayi": "Did the user agree to participate in the survey? Answer only true or false.",
        "yas_araligi": "What is the user's age or age range?",
        "egitim_durumu": "What is the user's education level?",
        "calisma_durumu_ve_meslek": "What is the user's employment status and profession?",
        "aylik_gelir": "What is the user's monthly income level?",
        "en_onemli_toplumsal_sorun": "According to the user, what is the most important social issue in Turkey?",
        "en_onemli_yerel_sorun": "According to the user, what is the most important local issue in their city?",
        "son_secim_oy_tercihi": "Which political party did the user vote for in the last election?",
        "bu_pazar_secim_oy_tercihi": "If an election were held this Sunday, which party would the user vote for?",
        "ek_gorusler": "Does the user have any additional wishes or suggestions?"
    }

    #  Fetch unprocessed call logs
    print("⬇️ Fetching unprocessed call logs...")
    try:
        response = supabase.from_('call_logs').select('id, transcript').eq('is_classified', False).execute()
    except Exception as e:
        print(f"❌ DATABASE ERROR: Could not fetch call logs. Details: {e}")
        return

    if not response.data:
        print("✅ No new calls to process.")
        return

    unprocessed_logs = response.data
    print(f"📊 Found {len(unprocessed_logs)} calls to process.")

    # Loop through each log
    for log in unprocessed_logs:
        call_id = log['id']
        transcript = log['transcript']
        print(f"\nProcessing Call ID: {call_id}...")

        if not transcript or len(transcript.strip()) < 20:
            print("   Skipping call with empty or short transcript.")
            supabase.from_('call_logs').update({'is_classified': True}).eq('id', call_id).execute()
            continue

        # 4. For each call, loop through the questions and get answers one by one
        survey_results = {"call_log_id": call_id}
        for db_column, question_text in questions_to_ask.items():
            print(f"   🔎 Asking AI for: '{db_column}'...")
            answer = find_answer_for_question(transcript, question_text)

            # Special handling for boolean
            if db_column == "katilim_onayi":
                survey_results[db_column] = 'true' in answer.lower()
            else:
                survey_results[db_column] = answer

            print(f"    AI Answer: {answer}")

        #  Save all collected answers to the database
        try:
            print(f"   💾 Saving collected results for call {call_id} to the database...")
            supabase.from_('survey_results').insert(survey_results).execute()

            # Mark the original log as classified
            supabase.from_('call_logs').update({'is_classified': True}).eq('id', call_id).execute()
            print(f"    Successfully processed and saved call {call_id}.")
        except Exception as e:
            print(f"    DATABASE ERROR: Could not save results for call {call_id}.")
            print(f"   Details: {e}")
            # Mark as classified to avoid re-processing a failed entry
            supabase.from_('call_logs').update({'is_classified': True}).eq('id', call_id).execute()

    print(f"\n Process Complete!  ")

# --- Run the main function ---
process_unclassified_calls()
