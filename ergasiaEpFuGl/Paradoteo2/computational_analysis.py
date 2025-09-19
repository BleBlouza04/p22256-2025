import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# --- A: 2 προτάσεις ---
sentences_A = {
    "S1": {
        "original": "Hope you too, to enjoy it as my deepest wishes.",
        "reconstructed": "I hope you too, enjoy it as my deepest wishes."
    },
    "S2": {
        "original": "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation.",
        "reconstructed": "Anyway, I believe the team, although a bit late and less communication at recent days, they really tried best for paper and cooperation."
    }
}

# --- B: ολόκληρα κείμενα ---
texts_B = {
    "Text1": {
        "original": "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication",
        "gpt2": "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication. Thank you very much.",
        "t5": "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. I got this message to see the approved message. In fact, I received the message from the professor, to show me, this, a couple of days ago.",
        "languagetool": "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication"
    },
    "Text2": {
        "original": "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets",
        "gpt2": "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets. Thank you all for your support, and I hope to see you in the future.",
        "t5": "During our final discuss, I told him about the new submission — the one we were waiting for since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although there was a bit delay and less communication at recent days, really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me if the doctor still",
        "languagetool": "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates were confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although a bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plans for the acknowledgments section edit before he's sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"
    }
}


# Φόρτωση μοντέλου embeddings
print("🔹 Loading embedding model (BERT MiniLM)...")
model = SentenceTransformer('all-MiniLM-L6-v2')


# Similarity για το A
print("\n-Cosine Similarity: A (Προτάσεις)")
embeddings_A = []
labels_A = []
for key, pair in sentences_A.items():
    emb_orig = model.encode([pair["original"]])
    emb_recon = model.encode([pair["reconstructed"]])
    sim = cosine_similarity(emb_orig, emb_recon)[0][0]
    print(f"{key}: {sim:.4f}")

    embeddings_A.extend([emb_orig[0], emb_recon[0]])
    labels_A.extend([f"{key} Original", f"{key} Reconstructed"])


# Similarity για το B
print("\n-Cosine Similarity: B (Κείμενα)")
embeddings_B = []
labels_B = []
for text_name, versions in texts_B.items():
    emb_orig = model.encode([versions["original"]])
    embeddings_B.append(emb_orig[0])
    labels_B.append(f"{text_name} Original")

    for method in ["gpt2", "t5", "languagetool"]:
        emb_recon = model.encode([versions[method]])
        sim = cosine_similarity(emb_orig, emb_recon)[0][0]
        print(f"{text_name} vs {method.upper()}: {sim:.4f}")

        embeddings_B.append(emb_recon[0])
        labels_B.append(f"{text_name} {method.upper()}")


# Οπτικοποίηση με PCA και t-SNE
print("\n-Οπτικοποιήσεις PCA/t-SNE για A και B")

def visualize_embeddings(embeddings, labels, title_prefix="A"):
    # PCA
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(embeddings)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=max(2, len(embeddings)//2), random_state=42)
    coords_tsne = tsne.fit_transform(embeddings)

    # -------- Πλοκή PCA --------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for i, label in enumerate(labels):
        color = 'blue' if "Original" in label else 'green'
        plt.scatter(coords_pca[i, 0], coords_pca[i, 1], color=color, label=label if i < 2 else "")
        plt.text(coords_pca[i, 0]+0.01, coords_pca[i, 1]+0.01, label, fontsize=8)
    plt.title(f"{title_prefix}: PCA of Embeddings")
    plt.legend()

    # -------- Πλοκή t-SNE --------
    plt.subplot(1, 2, 2)
    for i, label in enumerate(labels):
        color = 'blue' if "Original" in label else 'green'
        plt.scatter(coords_tsne[i, 0], coords_tsne[i, 1], color=color, label=label if i < 2 else "")
        plt.text(coords_tsne[i, 0]+5, coords_tsne[i, 1]+5, label, fontsize=8)
    plt.title(f"{title_prefix}: t-SNE of Embeddings")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Οπτικοποιήσεις
visualize_embeddings(np.array(embeddings_A), labels_A, title_prefix="A")
visualize_embeddings(np.array(embeddings_B), labels_B, title_prefix="B")