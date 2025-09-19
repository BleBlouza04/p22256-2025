from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# ---------------- Setup device ----------------
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to {'cuda' if device == 0 else 'cpu'}")

# ---------------- Load models ----------------
# Grammar correction
t5_model_name = "vennify/t5-base-grammar-correction"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
t5_pipeline = pipeline("text2text-generation", model=t5_model, tokenizer=t5_tokenizer, device=device)

# Paraphraser
paraphrase_model_name = "ramsrigouthamg/t5_paraphraser"
paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name)
paraphrase_pipeline = pipeline("text2text-generation", model=paraphrase_model, tokenizer=paraphrase_tokenizer, device=device)

# ---------------- Functions ----------------
def correct_text_t5(text):
    input_text = "grammar: " + text.strip().replace("\n", " ")
    result = t5_pipeline(input_text, max_length=256, num_return_sequences=1)
    return result[0]['generated_text']

def paraphrase_text(text, n=3):
    input_text = "paraphrase: " + text.strip().replace("\n", " ")
    result = paraphrase_pipeline(
        input_text,
        max_length=100,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        num_return_sequences=n
    )
    return [r['generated_text'] for r in result]

def reconstruct_text(text):
    corrected = correct_text_t5(text)
    paraphrased_list = paraphrase_text(corrected, n=3)
    return corrected, paraphrased_list

# ---------------- Sentences (Παραδοτέο 1.A) ----------------
sentence_1 = "Hope you too, to enjoy it as my deepest wishes."
sentence_2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

print("=== Πρόταση 1 ===")
c1, p1_list = reconstruct_text(sentence_1)
print("Original:", sentence_1)
print("Corrected:", c1)
print("Paraphrased options:")
for i, p in enumerate(p1_list, 1):
    print(f"  {i}. {p}")

print("\n=== Πρόταση 2 ===")
c2, p2_list = reconstruct_text(sentence_2)
print("Original:", sentence_2)
print("Corrected:", c2)
print("Paraphrased options:")
for i, p in enumerate(p2_list, 1):
    print(f"  {i}. {p}")
