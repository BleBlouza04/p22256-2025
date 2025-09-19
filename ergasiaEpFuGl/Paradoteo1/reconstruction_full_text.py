import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import language_tool_python


# ------------- Pipeline 1: GPT-style model -------------
def reconstruct_with_gpt(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=512,
        num_beams=5,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ------------- Pipeline 2: T5 paraphrasing -------------
def reconstruct_with_t5(text):
    tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")

    input_text = "paraphrase: " + text.strip().replace("\n", " ")
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ------------- Pipeline 3: Grammar correction with LanguageTool -------------
def reconstruct_with_languagetool(text, tool):
    return tool.correct(text)


# ------------- File helpers -------------
def load_text(filename):
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, filename)
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_text(filename, text):
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(text)


# ------------- Main logic -------------
def main():
    tool = language_tool_python.LanguageTool('en-US')

    texts = {
        "keimeno1": {
            "input": "keimeno1.txt",
            "output": "keimeno1_corrected.txt"
        },
        "keimeno2": {
            "input": "keimeno2.txt",
            "output": "keimeno2_corrected.txt"
        }
    }

    for name, paths in texts.items():
        input_text = load_text(paths["input"])

        results = []

        print(f"\n--- [{name}] with GPT2 ---")
        try:
            gpt_output = reconstruct_with_gpt(input_text)
            print(gpt_output)
            results.append("=== GPT2 Output ===\n" + gpt_output + "\n")
        except Exception as e:
            err = f"[GPT2 Error] {e}"
            print(err)
            results.append("=== GPT2 Output ===\n" + err + "\n")

        print(f"\n--- [{name}] with T5 ---")
        try:
            t5_output = reconstruct_with_t5(input_text)
            print(t5_output)
            results.append("=== T5 Output ===\n" + t5_output + "\n")
        except Exception as e:
            err = f"[T5 Error] {e}"
            print(err)
            results.append("=== T5 Output ===\n" + err + "\n")

        print(f"\n--- [{name}] with LanguageTool ---")
        try:
            lt_output = reconstruct_with_languagetool(input_text, tool)
            print(lt_output)
            results.append("=== LanguageTool Output ===\n" + lt_output + "\n")
        except Exception as e:
            err = f"[LT Error] {e}"
            print(err)
            results.append("=== LanguageTool Output ===\n" + err + "\n")

        # Αποθήκευση στο αρχείο
        save_text(paths["output"], "\n".join(results))

        print(f"Το '{paths['output']}' δημιουργήθηκε επιτυχώς με όλα τα αποτελέσματα.")

    tool.close()


if __name__ == "__main__":
    main()
