import re
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# --------- Load helper ---------
def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip()

def extract_outputs(corrected_text):
    """Κάνει parsing το consolidated αρχείο και επιστρέφει dict με τα outputs"""
    sections = re.split(r"=== (.*?) Output ===", corrected_text)
    results = {}
    for i in range(1, len(sections), 2):
        key = sections[i].strip()
        value = sections[i+1].strip()
        results[key] = value
    return results

# --------- Evaluation functions ---------
def compute_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

def compute_bertscore(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
    return float(F1[0])

# --------- Main ---------
def main():
    texts = {
        "keimeno1": {
            "original": "keimeno1.txt",
            "corrected": "keimeno1_corrected.txt"
        },
        "keimeno2": {
            "original": "keimeno2.txt",
            "corrected": "keimeno2_corrected.txt"
        }
    }

    for name, paths in texts.items():
        print(f"\n=== Αξιολόγηση {name} ===")

        reference = load_text(paths["original"])
        corrected = load_text(paths["corrected"])
        outputs = extract_outputs(corrected)

        for method, candidate in outputs.items():
            bleu = compute_bleu(reference, candidate)
            rouge = compute_rouge(reference, candidate)
            bert = compute_bertscore(reference, candidate)

            print(f"\n--- {method} ---")
            print(f"BLEU: {bleu:.4f}")
            print(f"ROUGE-1: {rouge['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-L: {rouge['rougeL'].fmeasure:.4f}")
            print(f"BERTScore-F1: {bert:.4f}")

if __name__ == "__main__":
    main()
