"""
Word Embeddings Explorer
  Part i  : Pretrained word vectors + vector arithmetic (analogies)
  Part ii : PCA / t-SNE visualisation of a 10-word domain (sports),
            cluster analysis, and a semantically-similar-word generator.

Install:  pip install gensim scikit-learn matplotlib numpy
Run:      python word_embeddings_explorer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------
# 0. Load pretrained vectors (GloVe, 100-d, trained on Wikipedia+Gigaword)
# ----------------------------------------------------------------------
print("Loading pretrained GloVe vectors (first run downloads ~130 MB)...")
model = api.load("glove-wiki-gigaword-100")
print(f"Vocabulary size: {len(model.key_to_index):,} | Dimensions: {model.vector_size}\n")


# ----------------------------------------------------------------------
# PART i: Vector arithmetic
# ----------------------------------------------------------------------
def solve_analogy(a, b, c, topn=3):
    """a is to b as c is to ?   ->  vec(b) - vec(a) + vec(c)"""
    result = model.most_similar(positive=[b, c], negative=[a], topn=topn)
    return result


def manual_arithmetic(positive, negative, topn=5):
    """Explicit numpy arithmetic on vectors (same idea, shown step by step)."""
    vec = sum(model[w] for w in positive) - sum(model[w] for w in negative)
    candidates = model.similar_by_vector(vec, topn=topn + len(positive) + len(negative))
    exclude = set(positive) | set(negative)
    return [(w, s) for w, s in candidates if w not in exclude][:topn]


def part_one():
    print("=" * 70)
    print("PART i: WORD RELATIONSHIPS VIA VECTOR ARITHMETIC")
    print("=" * 70)

    # 1. Basic similarity
    print("\n[1] Cosine similarity between word pairs")
    for w1, w2 in [("king", "queen"), ("cricket", "football"), ("cricket", "banana"),
                   ("laptop", "computer"), ("doctor", "nurse")]:
        print(f"   similarity({w1}, {w2}) = {model.similarity(w1, w2):.3f}")

    # 2. Classic analogies: a : b :: c : ?
    print("\n[2] Analogies   (a : b :: c : ?)   computed as  b - a + c")
    analogies = [
        ("man", "king", "woman"),           # gender / royalty
        ("france", "paris", "india"),       # country -> capital
        ("walk", "walked", "swim"),         # verb tense
        ("good", "better", "bad"),          # comparative
        ("japan", "yen", "india"),          # country -> currency
        ("messi", "football", "federer"),   # person -> sport
    ]
    for a, b, c in analogies:
        try:
            res = solve_analogy(a, b, c)
            best = ", ".join(f"{w} ({s:.2f})" for w, s in res)
            print(f"   {a} : {b} :: {c} : ?   ->  {best}")
        except KeyError as e:
            print(f"   Skipped ({e})")

    # 3. Manual arithmetic
    print("\n[3] Explicit vector arithmetic")
    tests = [
        (["king", "woman"], ["man"]),        # king - man + woman
        (["paris", "italy"], ["france"]),    # paris - france + italy
        (["computer", "sports"], []),        # blending two concepts
    ]
    for pos, neg in tests:
        expr = " + ".join(pos) + ("".join(f" - {n}" for n in neg))
        out = manual_arithmetic(pos, neg)
        print(f"   {expr}  ->  " + ", ".join(f"{w} ({s:.2f})" for w, s in out))

    # 4. Odd-one-out
    print("\n[4] Odd one out")
    for group in (["apple", "banana", "mango", "laptop"],
                  ["cricket", "tennis", "hockey", "python"]):
        print(f"   {group} -> {model.doesnt_match(group)}")

    print("\nAnalysis: Relationships are encoded as roughly consistent directions in "
          "vector space (e.g. gender, capital-of, tense). Results are approximate: "
          "the correct answer is usually top-ranked but not always, and biases in "
          "the training text (Wikipedia/news) can leak into results.\n")


# ----------------------------------------------------------------------
# PART ii-a: Dimensionality reduction on a 10-word domain
# ----------------------------------------------------------------------
SPORTS_WORDS = ["football", "cricket", "tennis", "basketball", "baseball",
                "hockey", "golf", "rugby", "athlete", "stadium"]


def visualise(words, title_prefix="Sports"):
    vectors = np.array([model[w] for w in words])

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(vectors)

    # t-SNE (perplexity must be < number of samples)
    tsne = TSNE(n_components=2, perplexity=5, init="pca",
                learning_rate="auto", random_state=42)
    tsne_2d = tsne.fit_transform(vectors)

    # Clustering to support the analysis
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(vectors)
    labels = km.labels_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, name in [(axes[0], pca_2d, f"PCA (explained var: {pca.explained_variance_ratio_.sum():.0%})"),
                             (axes[1], tsne_2d, "t-SNE")]:
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=140, alpha=0.85)
        for i, w in enumerate(words):
            ax.annotate(w, (coords[i, 0], coords[i, 1]), xytext=(6, 6),
                        textcoords="offset points", fontsize=11)
        ax.set_title(f"{title_prefix} embeddings - {name}")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("sports_embeddings.png", dpi=150)
    print("Saved plot -> sports_embeddings.png")
    plt.show()

    return labels


def analyse_clusters(words, labels):
    print("\nK-Means clusters (k=3):")
    for k in sorted(set(labels)):
        print(f"   Cluster {k}: {[w for w, l in zip(words, labels) if l == k]}")

    # Closest pair / most distant pair using cosine similarity
    pairs = [(words[i], words[j], model.similarity(words[i], words[j]))
             for i in range(len(words)) for j in range(i + 1, len(words))]
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nMost similar pairs :", [(a, b, round(s, 2)) for a, b, s in pairs[:3]])
    print("Least similar pairs:", [(a, b, round(s, 2)) for a, b, s in pairs[-3:]])

    print("\nAnalysis: Ball/team sports (football, basketball, baseball, rugby, "
          "hockey, cricket) sit close together, while the general terms 'athlete' "
          "and 'stadium' lie apart because they describe people/places rather than "
          "specific games. Individual sports such as golf and tennis usually form "
          "their own neighbourhood. PCA preserves global variance and is "
          "deterministic; t-SNE emphasises local neighbourhoods, so cluster shapes "
          "look tighter but distances between clusters are less meaningful.\n")


def part_two_visualisation():
    print("=" * 70)
    print("PART ii: VISUALISING EMBEDDINGS (domain = sports)")
    print("=" * 70)
    labels = visualise(SPORTS_WORDS)
    analyse_clusters(SPORTS_WORDS, labels)


# ----------------------------------------------------------------------
# PART ii-b: Semantically similar word generator
# ----------------------------------------------------------------------
def is_variant(a, b):
    """Filter trivial morphological variants (e.g. 'run' vs 'running')."""
    return a in b or b in a


def get_similar_words(word, n=5, skip_variants=True):
    """Return n semantically similar words with cosine similarity scores."""
    word = word.lower().strip()
    if word not in model.key_to_index:
        return f"'{word}' is not in the vocabulary."
    candidates = model.most_similar(word, topn=n * 6)
    results = []
    for w, score in candidates:
        if skip_variants and is_variant(word, w):
            continue
        if not w.isalpha():
            continue
        results.append((w, round(float(score), 3)))
        if len(results) == n:
            break
    return results


def sentence_vector(sentence):
    """Average the vectors of known words: a simple contextual representation."""
    tokens = [t for t in sentence.lower().split() if t in model.key_to_index]
    if not tokens:
        return None, []
    return np.mean([model[t] for t in tokens], axis=0), tokens


def context_related_words(sentence, n=5):
    """Words related to the *whole* sentence (context), not just one word."""
    vec, tokens = sentence_vector(sentence)
    if vec is None:
        return "No known words in sentence."
    cands = model.similar_by_vector(vec, topn=n + len(tokens) + 5)
    return [(w, round(float(s), 3)) for w, s in cands if w not in tokens][:n]


def part_two_generator():
    print("=" * 70)
    print("SEMANTICALLY SIMILAR WORD GENERATOR")
    print("=" * 70)

    demo = ["cricket", "computer", "happy", "doctor", "london"]
    for w in demo:
        print(f"\n5 words similar to '{w}':")
        for sim_word, score in get_similar_words(w):
            print(f"   {sim_word:<15} similarity = {score}")

    print("\nContext-aware related words (average of sentence vectors):")
    for s in ["the striker scored in the final match",
              "the neural network was trained on large data"]:
        print(f"   \"{s}\" -> {context_related_words(s)}")

    # Interactive mode
    print("\nInteractive mode (type 'quit' to exit)")
    while True:
        user = input("Enter a word: ").strip()
        if user.lower() in {"quit", "exit", "q", ""}:
            break
        result = get_similar_words(user)
        if isinstance(result, str):
            print("  ", result)
        else:
            for w, s in result:
                print(f"   {w:<15} {s}")


if __name__ == "__main__":
    part_one()
    part_two_visualisation()
    part_two_generator()
