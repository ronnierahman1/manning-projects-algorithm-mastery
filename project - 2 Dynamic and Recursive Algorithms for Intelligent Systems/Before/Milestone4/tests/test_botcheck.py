import time
import json
import os
import sys
from difflib import SequenceMatcher

# Add parent directory to sys.path so 'chatbot' and 'ai' modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot.chatbot import AIChatbot
from chatbot.knowledge_base import KnowledgeBase
from ai.ai_module import AIModule
import logging

# Ensure the log directory exists
output_dir = "output"
output_file = os.path.join(output_dir, "output.log")
os.makedirs(output_dir, exist_ok=True)

# Truncate (overwrite) the log file before writing new logs
with open(output_file, "w", encoding="utf-8"):
    pass  # This truncates the file
kb_path = "data/dev-v2.0.json"
# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(output_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def log(msg):
    logging.info(msg)

# Load dev data (ground truth)
with open(kb_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

# Build lookup dictionary: question → list of answers
qa_lookup = {}
for item in dev_data["data"]:
    for para in item["paragraphs"]:
        for qa in para["qas"]:
            if qa.get("is_impossible", False):
                continue
            question = qa["question"].strip().rstrip(".?!")
            answers = [a["text"] for a in qa["answers"] if a.get("text")]
            if question and answers:
                qa_lookup[question] = answers

# Load chatbot

kb = KnowledgeBase(kb_path)
chatbot = AIChatbot(kb_path)

# Metrics
correct = 0
incorrect = 0

# Test chatbot on known questions
log("🤖 Initializing Recursive AI Chatbot...\n" + "=" * 50)
log(f"✅ Loaded {kb.size()} QA pairs from knowledge base.")
log("Chatbot initialized successfully!\n")
count = 0 
exactmatch_count = 0
fuzzy_match_count = 0 
max_processing_time = 0.0
min_processing_time = float('inf')
average_processing_time = 0.0
total_processing_time = 0.0
for idx, (question, correct_answers) in enumerate(qa_lookup.items(), start=1):
    fuzzy_match = False
    count += 1
    log(f"Q{idx}: {question}")
    start_time = time.perf_counter()
    predicted, is_fuzzy, threshold = chatbot.handle_query(question)
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    if duration_ms > max_processing_time:
        max_processing_time = duration_ms
    if duration_ms < min_processing_time:
        min_processing_time = duration_ms
    predicted_clean = predicted.lower().strip().rstrip(".?!")
    expected_clean = [ans.lower().strip().rstrip(".?!") for ans in correct_answers]
    total_processing_time += duration_ms

    # First try substring match
    matched = any(expected in predicted_clean or predicted_clean in expected for expected in expected_clean)
    
    # If no substring match, try fuzzy match with progressive thresholds
    if not matched:
        for threshold in [0.95, 0.90, 0.85,0.8, 0.75, 0.7, 0.65, 0.6]:
            matched = any(
                SequenceMatcher(None, predicted_clean, expected).ratio() >= threshold
                for expected in expected_clean
            )
            if matched:
                fuzzy_match = True
                log(f"Fuzzy match at threshold {threshold:.2f}")
                break    # if any(expected in predicted_clean for expected in expected_clean):
    if matched:
        log(f"A{idx}: ✅ {predicted}")
        correct += 1
        if (fuzzy_match == True):
            fuzzy_match_count += 1
        else:
            exactmatch_count +=1
    else:
        log(f"A{idx}: ❌ {predicted}")
        log(f"Expected: {correct_answers}")
        incorrect += 1
    log(f"⏱️ Answered in {duration_ms:.2f} ms\n" + "-" * 80)

# Summary
average_processing_time = total_processing_time / count if count > 0 else 0.
log("\n📊 Summary")
log("=" * 40)
log(f"✅ Correct:   {correct}")
log(f"❌ Incorrect: {incorrect}")
log(f"🎯 Accuracy:  {correct / (correct + incorrect):.2%}")
log(f"Exact matches: {exactmatch_count}")
log(f"Fuzzy matches: {fuzzy_match_count}")
log(f"Total queries processed: {correct + incorrect}")
log(f"Max processing time: {max_processing_time:.2f} ms")
log(f"Min processing time: {min_processing_time:.2f} ms")
log(f"Average processing time: {average_processing_time:.2f} ms")
log(f"Total processing time: {total_processing_time:.2f} ms")
