Paradoteo1/
  ├── reconstruction_phrases.py     # Παραδοτέο 1Α: Aνακατασκευή  2 προτάσεων
  ├── reconstruction_full_text.py   # Παραδοτέο 1Β: Ανακατασκευή των 2 κειμένων με GPT-2, T5, LanguageTool
  ├── results_comparison.py         # Παραδοτέο 1C: Σύγκριση αποτελεσμάτων
  ├── keimeno1.txt 
  ├── keimeno2.txt
  ├── keimeno1_corrected.txt
  └── keimeno2_corrected.txt
  
Paradoteo2/
  └── computational_analysis.py     # Παραδοτέο 2

## Commands
conda activate nlp-recon
python reconstruction_phrases.py
python reconstruction_full_text.py
python results_comparison.py
python computational_analysis.py

## Dependencies

The project requires the following libraries:

```bash
pip install torch transformers sentence-transformers language-tool-python \
            nltk rouge-score bert-score matplotlib scikit-learn

