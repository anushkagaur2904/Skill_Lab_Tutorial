"""
Domain-specific Word2Vec on a small legal + medical corpus
==========================================================

* Word2Vec (skip-gram, negative sampling) implemented from scratch in NumPy
  (gensim is not required; a gensim equivalent is shown at the bottom).
* Two small synthetic corpora (legal / medical) with *polysemous* words that
  mean different things in each domain: discharge, trial, case, complaint,
  condition, motion, order.
* Analyses:
    1. Nearest neighbours of ambiguous words in domain-only vs. joint models
    2. Category coherence (do drugs cluster with drugs, courts with courts...?)
    3. Analogies across domains (physician:patient :: attorney:?)
    4. Domain-bias score: is a polysemous word pulled toward one domain?
    5. Learning curve: how much text is needed?
    6. t-SNE map of the joint embedding space

Run:  python domain_word2vec.py
"""
import re
import random
import sys
from collections import Counter

import numpy as np

# =============================================================================
# 1. Synthetic domain corpora
# =============================================================================
# A template-based generator keeps the corpus small, controlled and reproducible.
# Words in the same slot share contexts, which is exactly the signal Word2Vec
# uses (distributional hypothesis).

LEGAL_SLOTS = {
    "party":  ["plaintiff", "defendant", "appellant", "respondent"],
    "lawyer": ["attorney", "counsel"],
    "court":  ["court", "tribunal", "appellate_court", "district_court"],
    "doc":    ["contract", "statute", "lease", "agreement", "deed", "warranty", "treaty"],
    "filing": ["petition", "brief", "affidavit", "summons", "subpoena", "appeal"],
    "claim":  ["negligence", "fraud", "defamation", "trespass", "libel", "breach"],
    "remedy": ["damages", "injunction", "restitution", "settlement", "sanctions"],
    "ruling": ["dismissed", "granted", "denied", "upheld", "overturned"],
    "proc":   ["hearing", "arbitration", "deposition", "litigation"],
}
LEGAL_PLAIN = [
    "the {party} filed a {filing} in the {court}",
    "the {lawyer} represented the client in the {proc}",
    "the {court} {ruling} the {filing} filed by the {party}",
    "the {party} alleged {claim} and sought {remedy}",
    "the {court} awarded {remedy} to the {party} for {claim}",
    "the {lawyer} argued that the {doc} was void under the statute",
    "the {party} breached the {doc} and the {court} awarded {remedy}",
    "the judge {ruling} the {filing} after the {proc}",
    "the witness gave testimony during the {proc} before the {court}",
    "the {lawyer} advised the client to accept the {remedy}",
    "the {court} interpreted the statute and the {doc}",
    "the {party} appeared before the {court}",
    "the {lawyer} cross examined the witness during the {proc}",
    "the judge instructed the jury on {claim} and {remedy}",
    "the {court} held that the {doc} was enforceable",
]
LEGAL_POLY = {
    "discharge": [
        "the bankruptcy {court} approved the discharge of the debtor from all {claim} liability",
        "the {court} granted a discharge of the debtor obligations under the {doc}",
        "the {party} sought a discharge of the lien after the {proc}",
    ],
    "trial": [
        "the {lawyer} prepared the witness for trial before the jury",
        "the {court} scheduled the trial after the {proc}",
        "the jury returned a verdict at trial in favor of the {party}",
    ],
    "case": [
        "the {lawyer} argued the case before the {court}",
        "the judge dismissed the case after the {proc}",
        "the {party} lost the case and filed an appeal",
    ],
    "complaint": [
        "the {party} filed a complaint in the {court} alleging {claim}",
        "the {court} dismissed the complaint for lack of jurisdiction",
        "the {lawyer} amended the complaint to add a claim of {claim}",
    ],
    "condition": [
        "the {doc} contained a condition precedent to performance",
        "the {court} held that the condition was satisfied under the {doc}",
        "the {party} failed to meet the condition and breached the {doc}",
    ],
    "motion": [
        "the {lawyer} filed a motion to dismiss the complaint",
        "the {court} granted the motion for summary judgment",
        "the {party} opposed the motion before the {court}",
    ],
    "order": [
        "the {court} issued an order granting the {remedy}",
        "the {party} violated the order and the {court} imposed {remedy}",
        "the judge signed the order after the {proc}",
    ],
}

MED_SLOTS = {
    "clinician": ["physician", "surgeon", "nurse", "clinician", "pharmacist", "radiologist"],
    "symptom":   ["fever", "cough", "nausea", "fatigue", "headache", "dyspnea", "rash", "dizziness"],
    "disease":   ["diabetes", "pneumonia", "hypertension", "asthma", "cancer", "sepsis", "anemia", "arthritis"],
    "drug":      ["insulin", "metformin", "aspirin", "ibuprofen", "amoxicillin", "warfarin", "morphine"],
    "test":      ["biopsy", "mri", "ultrasound", "endoscopy", "bloodwork", "xray"],
    "finding":   ["tumor", "lesion", "infection", "inflammation", "fracture", "clot"],
    "proc":      ["surgery", "chemotherapy", "dialysis", "transplant", "radiotherapy", "angioplasty"],
    "place":     ["hospital", "clinic", "ward", "icu"],
}
MED_PLAIN = [
    "the {clinician} treated the patient for {disease}",
    "the patient presented with {symptom} and {symptom}",
    "the {clinician} prescribed {drug} for {disease}",
    "{drug} is used to treat {disease}",
    "the {test} revealed a {finding} in the patient",
    "the patient underwent {proc} at the {place}",
    "the patient was admitted to the {place} with {symptom}",
    "the {clinician} ordered a {test} to rule out {disease}",
    "the patient reported {symptom} after taking {drug}",
    "the {clinician} monitored the patient after {proc}",
    "{disease} can cause {symptom} and {symptom}",
    "the {clinician} administered {drug} to the patient during {proc}",
    "the patient recovered from {disease} after {proc}",
    "the {clinician} diagnosed {disease} based on the {test}",
    "the {clinician} examined the patient in the {place}",
]
MED_POLY = {
    "discharge": [
        "the {clinician} approved the discharge of the patient from the {place}",
        "the patient was ready for discharge from the {place} after {proc}",
        "the patient had purulent discharge from the wound and {symptom}",
    ],
    "trial": [
        "the randomized trial tested {drug} in patients with {disease}",
        "the trial evaluated {drug} against placebo in patients with {disease}",
        "the trial reported fewer {symptom} in the {drug} group",
    ],
    "case": [
        "the {clinician} reported a case of {disease} with {symptom}",
        "the case report described {disease} after {proc}",
        "the case involved a patient with {disease} and {finding}",
    ],
    "complaint": [
        "the chief complaint of the patient was {symptom}",
        "the patient reported a complaint of {symptom} and {symptom}",
        "the {clinician} recorded the complaint and examined the patient",
    ],
    "condition": [
        "the condition of the patient improved after {drug}",
        "the {clinician} monitored the condition of the patient in the {place}",
        "the patient had a chronic condition such as {disease}",
    ],
    "motion": [
        "the patient had limited range of motion after the {finding}",
        "the {clinician} measured the range of motion in the joint after {proc}",
        "the patient regained motion in the joint after {proc}",
    ],
    "order": [
        "the {clinician} wrote an order for {test} and {drug}",
        "the nurse followed the order to give {drug} to the patient",
        "the {clinician} signed the order for {proc} at the {place}",
    ],
}
POLY_WORDS = list(LEGAL_POLY)
STOPWORDS = set("the a an of in to and for on at by with from after during before that was is as such all "
                "under into against".split())


def _fill(template, slots, rng):
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(slots[m.group(1)]), template).split()


def generate_corpus(slots, plain, poly, n_sentences, poly_frac=0.35, seed=0):
    rng = random.Random(seed)
    poly_templates = [t for ts in poly.values() for t in ts]
    out = []
    for _ in range(n_sentences):
        t = rng.choice(poly_templates) if rng.random() < poly_frac else rng.choice(plain)
        out.append(_fill(t, slots, rng))
    return out


# =============================================================================
# 2. Word2Vec: skip-gram + negative sampling (NumPy)
# =============================================================================
class Word2Vec:
    def __init__(self, dim=50, window=4, negatives=5, epochs=40, lr=0.025,
                 min_count=2, subsample=1e-2, batch=64, seed=0):
        self.dim, self.window, self.K = dim, window, negatives
        self.epochs, self.lr0, self.min_count = epochs, lr, min_count
        self.subsample, self.batch = subsample, batch
        self.rng = np.random.default_rng(seed)

    # -- vocabulary ------------------------------------------------------------
    def _build_vocab(self, sentences):
        cnt = Counter(w for s in sentences for w in s)
        words = [w for w, c in cnt.most_common() if c >= self.min_count]
        self.itos = words
        self.stoi = {w: i for i, w in enumerate(words)}
        self.counts = np.array([cnt[w] for w in words], dtype=np.float64)
        f = self.counts / self.counts.sum()
        # word2vec subsampling keep-probability
        self.keep = np.minimum(1.0, np.sqrt(self.subsample / f) + self.subsample / f)
        p = self.counts ** 0.75
        self.neg_cdf = np.cumsum(p / p.sum())

    def _encode(self, sentences):
        return [np.array([self.stoi[w] for w in s if w in self.stoi], dtype=np.int64)
                for s in sentences]

    # -- (center, context) pairs for one epoch -------------------------------
    def _make_pairs(self, enc):
        toks, sids = [], []
        for i, s in enumerate(enc):
            k = self.rng.random(len(s)) < self.keep[s]      # frequent-word subsampling
            s = s[k]
            toks.append(s)
            sids.append(np.full(len(s), i))
        toks, sids = np.concatenate(toks), np.concatenate(sids)
        n = len(toks)
        b = self.rng.integers(1, self.window + 1, size=n)    # dynamic window
        C, O = [], []
        for off in range(1, self.window + 1):
            idx = np.arange(n - off)
            same = sids[idx] == sids[idx + off]
            fwd = same & (b[idx] >= off)                     # center=idx, ctx=idx+off
            bwd = same & (b[idx + off] >= off)               # center=idx+off, ctx=idx
            C += [toks[idx][fwd], toks[idx + off][bwd]]
            O += [toks[idx + off][fwd], toks[idx][bwd]]
        return np.concatenate(C), np.concatenate(O)

    def _sample_neg(self, shape):
        return np.searchsorted(self.neg_cdf, self.rng.random(shape)).clip(0, len(self.itos) - 1)

    # -- training -------------------------------------------------------------
    def fit(self, sentences, verbose=False):
        self._build_vocab(sentences)
        enc = self._encode(sentences)
        V, d = len(self.itos), self.dim
        self.W = ((self.rng.random((V, d)) - 0.5) / d)       # input vectors (the embeddings)
        self.U = np.zeros((V, d))                             # output vectors
        sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        self.loss_history = []
        total_steps, step = None, 0
        for ep in range(self.epochs):
            C, O = self._make_pairs(enc)
            perm = self.rng.permutation(len(C))
            C, O = C[perm], O[perm]
            if total_steps is None:
                total_steps = self.epochs * int(np.ceil(len(C) / self.batch))
            ep_loss, nb = 0.0, 0
            for s in range(0, len(C), self.batch):
                c, o = C[s:s + self.batch], O[s:s + self.batch]
                n = self._sample_neg((len(c), self.K))
                lr = self.lr0 * max(0.05, 1 - step / total_steps)
                v, up, un = self.W[c], self.U[o], self.U[n]               # (B,d) (B,d) (B,K,d)
                sp = sig(np.sum(v * up, 1))                              # (B,)
                sn = sig(np.einsum("bd,bkd->bk", v, un))                 # (B,K)
                gp, gn = sp - 1.0, sn                                    # dLoss/dscore
                dv = gp[:, None] * up + np.einsum("bk,bkd->bd", gn, un)
                np.add.at(self.U, o, -lr * gp[:, None] * v)
                np.add.at(self.U, n, -lr * gn[:, :, None] * v[:, None, :])
                np.add.at(self.W, c, -lr * dv)
                ep_loss += -np.mean(np.log(sp + 1e-10)) - np.mean(np.sum(np.log(1 - sn + 1e-10), 1))
                nb += 1
                step += 1
            self.loss_history.append(ep_loss / nb)
            if verbose and (ep % 10 == 0 or ep == self.epochs - 1):
                print(f"    epoch {ep + 1:3d}/{self.epochs}  loss {self.loss_history[-1]:.4f}")
        self.Wn = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-12)
        return self

    # -- query API (mirrors gensim's KeyedVectors) ------------------------------
    def __contains__(self, w): return w in self.stoi
    def vec(self, w): return self.Wn[self.stoi[w]]

    def most_similar(self, word, topn=6, content_only=True):
        sims = self.Wn @ self.vec(word)
        out = []
        for i in np.argsort(-sims):
            w = self.itos[i]
            if w == word or (content_only and w in STOPWORDS):
                continue
            out.append((w, float(sims[i])))
            if len(out) == topn:
                break
        return out

    def analogy(self, a, b, c, topn=3):
        """a : b :: c : ?   ->  vec(b) - vec(a) + vec(c)"""
        q = self.vec(b) - self.vec(a) + self.vec(c)
        q /= np.linalg.norm(q)
        sims = self.Wn @ q
        out = []
        for i in np.argsort(-sims):
            w = self.itos[i]
            if w in (a, b, c) or w in STOPWORDS:
                continue
            out.append((w, float(sims[i])))
            if len(out) == topn:
                break
        return out

    def save_text(self, path):
        with open(path, "w") as f:
            f.write(f"{len(self.itos)} {self.dim}\n")
            for w, v in zip(self.itos, self.W):
                f.write(w + " " + " ".join(f"{x:.5f}" for x in v) + "\n")


# =============================================================================
# 3. Analysis helpers
# =============================================================================
def category_map():
    cats = {}
    for name, ws in LEGAL_SLOTS.items():
        cats[f"legal:{name}"] = ws
    for name, ws in MED_SLOTS.items():
        cats[f"med:{name}"] = ws
    cats["legal:role"] = ["judge", "witness", "jury", "client"]
    cats["med:role"] = ["patient"]
    return cats


def word2cat(model):
    m = {}
    for c, ws in category_map().items():
        for w in ws:
            if w in model and w not in POLY_WORDS:
                m[w] = c
    return m


def purity(model, level="category", k=5):
    """Mean fraction of a word's k nearest labelled neighbours that share its label."""
    w2c = word2cat(model)
    if level == "domain":
        w2c = {w: c.split(":")[0] for w, c in w2c.items()}
    words = list(w2c)
    M = np.stack([model.vec(w) for w in words])
    S = M @ M.T
    np.fill_diagonal(S, -np.inf)
    scores = []
    for i, w in enumerate(words):
        nn = np.argsort(-S[i])[:k]
        scores.append(np.mean([w2c[words[j]] == w2c[w] for j in nn]))
    return float(np.mean(scores))


def domain_centroids(model):
    leg = [w for w in sum(LEGAL_SLOTS.values(), []) + ["judge", "witness", "jury", "client"]
           if w in model and w not in POLY_WORDS]
    med = [w for w in sum(MED_SLOTS.values(), []) + ["patient"]
           if w in model and w not in POLY_WORDS]
    cl = np.mean([model.vec(w) for w in leg], 0)
    cm = np.mean([model.vec(w) for w in med], 0)
    return leg, med, cl / np.linalg.norm(cl), cm / np.linalg.norm(cm)


def domain_bias(model, w, cl, cm):
    """+ => closer to medical centroid, - => closer to legal centroid."""
    v = model.vec(w)
    return float(v @ cm - v @ cl)


def fmt(nn): return ", ".join(f"{w} ({s:.2f})" for w, s in nn)


# =============================================================================
# 4. Main experiment
# =============================================================================
def main():
    N = 1500                                    # sentences per domain
    print("=" * 78)
    print("DOMAIN-SPECIFIC WORD2VEC: legal vs. medical")
    print("=" * 78)
    legal = generate_corpus(LEGAL_SLOTS, LEGAL_PLAIN, LEGAL_POLY, N, seed=1)
    medical = generate_corpus(MED_SLOTS, MED_PLAIN, MED_POLY, N, seed=2)
    joint = legal + medical
    random.Random(3).shuffle(joint)
    print(f"\nCorpus sizes: legal={sum(map(len, legal)):,} tokens, "
          f"medical={sum(map(len, medical)):,} tokens ({N} sentences each)")
    print("Example legal  :", " ".join(legal[0]))
    print("Example medical:", " ".join(medical[0]))

    hp = dict(dim=50, window=4, negatives=5, epochs=40, lr=0.025)
    print(f"\nHyperparameters: {hp}\n\nTraining models...")
    models = {}
    for name, corpus in [("legal", legal), ("medical", medical), ("joint", joint)]:
        print(f"  [{name}]")
        models[name] = Word2Vec(**hp, seed=7).fit(corpus, verbose=True)
        print(f"    vocab = {len(models[name].itos)}")
        models[name].save_text(f"vectors_{name}.txt")

    L, M, J = models["legal"], models["medical"], models["joint"]

    # ---- A. neighbours of ambiguous words -----------------------------------
    print("\n" + "=" * 78)
    print("A. SAME WORD, DIFFERENT DOMAIN: nearest neighbours (cosine)")
    print("=" * 78)
    for w in POLY_WORDS:
        print(f"\n'{w}'")
        print(f"   legal-only  : {fmt(L.most_similar(w, 5))}")
        print(f"   medical-only: {fmt(M.most_similar(w, 5))}")
        print(f"   joint model : {fmt(J.most_similar(w, 5))}")

    # ---- B. anchor similarity: 'complaint' etc. relative to domain anchors ---
    print("\n" + "=" * 78)
    print("B. SENSE ANCHORING: cosine to a domain anchor word in each domain-only model")
    print("=" * 78)
    print(f"{'word':<12}{'legal: sim(word, court)':>26}{'medical: sim(word, patient)':>30}")
    for w in POLY_WORDS:
        print(f"{w:<12}{float(L.vec(w) @ L.vec('court')):>26.3f}{float(M.vec(w) @ M.vec('patient')):>30.3f}")

    # ---- C. category coherence -----------------------------------------------
    print("\n" + "=" * 78)
    print("C. CATEGORY COHERENCE (nearest-5 label purity among unambiguous domain words)")
    print("=" * 78)
    for name, m in models.items():
        print(f"  {name:<8} category purity@5 = {purity(m, 'category'):.3f}"
              f"    domain purity@5 = {purity(m, 'domain'):.3f}")
    print("  (chance level for category purity is roughly 0.07-0.10)")

    leg, med, cl, cm = domain_centroids(J)
    Ml = np.stack([J.vec(w) for w in leg]); Mm = np.stack([J.vec(w) for w in med])
    within_l = (Ml @ Ml.T)[np.triu_indices(len(leg), 1)].mean()
    within_m = (Mm @ Mm.T)[np.triu_indices(len(med), 1)].mean()
    between = (Ml @ Mm.T).mean()
    print(f"\n  Joint model mean cosine: within-legal={within_l:.3f}  "
          f"within-medical={within_m:.3f}  legal<->medical={between:.3f}")

    # ---- D. analogies -------------------------------------------------------
    print("\n" + "=" * 78)
    print("D. ANALOGIES in the joint model  (a : b :: c : ?)")
    print("=" * 78)
    tests = [
        ("physician", "patient", "attorney", {"client"}),
        ("patient", "physician", "client", {"attorney", "counsel"}),
        ("hospital", "patient", "court", {"defendant", "plaintiff", "appellant", "respondent",
                                          "party", "judge", "jury", "witness"}),
        ("patient", "hospital", "defendant", {"court", "tribunal", "district_court", "appellate_court"}),
    ]
    hits = 0
    for a, b, c, gold in tests:
        res = J.analogy(a, b, c, 3)
        ok = res[0][0] in gold
        hits += ok
        print(f"  {a}:{b} :: {c}:?  ->  {fmt(res)}   {'OK' if ok else 'MISS'} (expected: {sorted(gold)[:3]}...)")
    print(f"  top-1 accuracy: {hits}/{len(tests)}")

    # ---- E. domain bias of polysemous words ---------------------------------
    print("\n" + "=" * 78)
    print("E. DOMAIN BIAS in the JOINT model   (bias = cos(w, medical centroid) - cos(w, legal centroid))")
    print("=" * 78)
    b_leg = [domain_bias(J, w, cl, cm) for w in leg]
    b_med = [domain_bias(J, w, cl, cm) for w in med]
    print(f"  unambiguous legal words   : mean {np.mean(b_leg):+.3f}  (n={len(b_leg)})")
    print(f"  unambiguous medical words : mean {np.mean(b_med):+.3f}  (n={len(b_med)})")
    poly_bias = {}
    for w in POLY_WORDS:
        poly_bias[w] = domain_bias(J, w, cl, cm)
        print(f"  {w:<12} {poly_bias[w]:+.3f}")
    print("  -> a single vector must represent both senses, so polysemous words have a bias near 0")
    print("     (they are pulled toward both domains) while unambiguous words are strongly one-sided.")

    # ---- F. learning curve --------------------------------------------------
    print("\n" + "=" * 78)
    print("F. HOW MUCH TEXT DO WE NEED?  (joint model, mean of 3 seeds)")
    print("=" * 78)
    sizes = [25, 50, 100, 200, 500, 1000, 1500]
    curve = {"n": sizes, "cat": [], "dom": []}
    print(f"  {'sent/domain':>12}{'category purity@5':>20}{'domain purity@5':>18}")
    for n in sizes:
        cs, ds = [], []
        for seed in range(3):
            c = (generate_corpus(LEGAL_SLOTS, LEGAL_PLAIN, LEGAL_POLY, n, seed=100 + seed)
                 + generate_corpus(MED_SLOTS, MED_PLAIN, MED_POLY, n, seed=200 + seed))
            random.Random(seed).shuffle(c)
            m = Word2Vec(**hp, seed=seed).fit(c)
            cs.append(purity(m, "category")); ds.append(purity(m, "domain"))
        curve["cat"].append(np.mean(cs)); curve["dom"].append(np.mean(ds))
        print(f"  {n:>12}{curve['cat'][-1]:>20.3f}{curve['dom'][-1]:>18.3f}")

    make_figures(J, models, curve, poly_bias, b_leg, b_med)
    print("\nSaved: embedding_map.png, analysis_plots.png, vectors_{legal,medical,joint}.txt")


# =============================================================================
# 5. Figures
# =============================================================================
def make_figures(J, models, curve, poly_bias, b_leg, b_med):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    w2c = word2cat(J)
    words = [w for w in J.itos if w in w2c or w in POLY_WORDS]
    X = np.stack([J.vec(w) for w in words])
    Z = TSNE(n_components=2, perplexity=12, metric="cosine", init="pca",
             random_state=0, max_iter=2000).fit_transform(X)

    cats = sorted(set(w2c.values()))
    cmap_l = plt.get_cmap("Blues"); cmap_m = plt.get_cmap("Reds")
    lc = [c for c in cats if c.startswith("legal")]; mc = [c for c in cats if c.startswith("med")]
    color = {c: cmap_l(0.35 + 0.6 * i / max(1, len(lc) - 1)) for i, c in enumerate(lc)}
    color.update({c: cmap_m(0.35 + 0.6 * i / max(1, len(mc) - 1)) for i, c in enumerate(mc)})

    fig, ax = plt.subplots(figsize=(13, 9))
    for i, w in enumerate(words):
        if w in POLY_WORDS:
            ax.scatter(*Z[i], s=260, marker="*", c="#2ca02c", edgecolor="k", zorder=5)
            ax.annotate(w, Z[i], xytext=(6, 6), textcoords="offset points",
                        fontsize=11, fontweight="bold", color="#1b6e1b")
        else:
            c = w2c[w]
            ax.scatter(*Z[i], s=45, c=[color[c]], marker="o" if c.startswith("legal") else "s",
                       edgecolor="k", linewidth=0.3)
            ax.annotate(w, Z[i], xytext=(3, 3), textcoords="offset points", fontsize=7, alpha=0.8)
    ax.scatter([], [], c=[cmap_l(0.6)], marker="o", label="legal vocabulary")
    ax.scatter([], [], c=[cmap_m(0.6)], marker="s", label="medical vocabulary")
    ax.scatter([], [], c="#2ca02c", marker="*", s=200, edgecolor="k", label="polysemous (both domains)")
    ax.legend(loc="best")
    ax.set_title("t-SNE of the joint Word2Vec model: legal and medical vocabulary form separate clusters")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig("embedding_map.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    ax.plot(curve["n"], curve["cat"], "o-", label="category purity@5")
    ax.plot(curve["n"], curve["dom"], "s-", label="domain purity@5")
    ax.set_xscale("log"); ax.set_xlabel("training sentences per domain"); ax.set_ylabel("purity")
    ax.set_title("Learning curve: small data is enough for coarse\nstructure, fine structure needs more"); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1]
    ws = sorted(poly_bias, key=poly_bias.get)
    ax.barh(ws, [poly_bias[w] for w in ws], color="#2ca02c")
    ax.axvline(np.mean(b_leg), color="tab:blue", ls="--", label=f"legal words mean ({np.mean(b_leg):+.2f})")
    ax.axvline(np.mean(b_med), color="tab:red", ls="--", label=f"medical words mean ({np.mean(b_med):+.2f})")
    ax.axvline(0, color="k", lw=.5)
    ax.set_xlabel("bias:  legal  <-  0  ->  medical"); ax.set_title("Polysemous words are far less one-sided\nthan unambiguous words (joint model)"); ax.legend(fontsize=8)

    ax = axes[2]
    for name, m in models.items():
        ax.plot(range(1, len(m.loss_history) + 1), m.loss_history, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("SGNS loss"); ax.set_title("Training loss"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("analysis_plots.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Equivalent gensim call (if you have gensim installed and a real corpus):
#
#   from gensim.models import Word2Vec
#   model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=4, sg=1,
#                    negative=5, min_count=2, epochs=40, sample=1e-2, seed=7)
#   model.wv.most_similar("complaint", topn=5)
#
# The vectors_*.txt files written above load directly with
#   KeyedVectors.load_word2vec_format("vectors_joint.txt")
# -----------------------------------------------------------------------------
