"""
Predict whether a student passes or fails using a TensorFlow/Keras DNN.

Usage
-----
  # Demo with synthetic data (study_hours + attendance)
  python student_performance_dnn.py

  # Your own CSV, default columns (study_hours, attendance, passed)
  python student_performance_dnn.py --csv students.csv

  # Your own CSV with extra features / different names
  python student_performance_dnn.py --csv students.csv \
      --features study_hours attendance prev_grade assignments_done \
      --target result

The target column may be 0/1 or text such as pass/fail, yes/no, true/false.
Features must be numeric. Rows with missing values are dropped.
"""

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def generate_data(n=2000):
    """Synthetic students; used only when no CSV is given."""
    study_hours = np.random.uniform(0, 10, n)
    attendance = np.random.uniform(40, 100, n)
    score = 0.55 * study_hours + 0.06 * attendance + np.random.normal(0, 0.8, n)
    return pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "passed": (score > 6.0).astype(int),
    })


def encode_target(series):
    """Turn a target column into 0/1 (1 = pass)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    positive = {"pass", "passed", "p", "yes", "y", "true", "1"}
    negative = {"fail", "failed", "f", "no", "n", "false", "0"}
    cleaned = series.astype(str).str.strip().str.lower()
    unknown = set(cleaned) - positive - negative
    if unknown:
        raise ValueError(f"Unrecognised target values: {sorted(unknown)}")
    return cleaned.isin(positive).astype(int)


def load_data(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        missing = [c for c in args.features + [args.target] if c not in df.columns]
        if missing:
            raise SystemExit(f"Columns not found in CSV: {missing}\n"
                             f"Available columns: {list(df.columns)}")
    else:
        print("No --csv given: using synthetic demo data.\n")
        df = generate_data()

    df = df[args.features + [args.target]].dropna()
    df[args.target] = encode_target(df[args.target])
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(n_features, lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),  # P(pass)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_results(history, model, scaler, X_test, y_test, features, path):
    two_d = len(features) == 2
    fig, axes = plt.subplots(1, 3 if two_d else 2, figsize=(17 if two_d else 11, 4.5))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch")
    axes[1].legend()

    if two_d:  # decision boundary only makes sense with two features
        X_orig = scaler.inverse_transform(X_test)
        x_rng = np.linspace(X_orig[:, 0].min(), X_orig[:, 0].max(), 200)
        y_rng = np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), 200)
        xx, yy = np.meshgrid(x_rng, y_rng)
        probs = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]),
                              verbose=0).reshape(xx.shape)
        cf = axes[2].contourf(xx, yy, probs, levels=20, cmap="RdYlGn", alpha=0.8)
        axes[2].contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=2)
        axes[2].scatter(X_orig[:, 0], X_orig[:, 1], c=y_test, cmap="RdYlGn",
                        edgecolors="k", s=18)
        axes[2].set(title="Decision boundary (black line = 50%)",
                    xlabel=features[0], ylabel=features[1])
        fig.colorbar(cf, ax=axes[2], label="P(pass)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved plots to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", help="Path to CSV file (omit for synthetic demo data)")
    p.add_argument("--features", nargs="+", default=["study_hours", "attendance"],
                   help="Numeric feature columns")
    p.add_argument("--target", default="passed", help="Pass/fail column")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="P(pass) cutoff; lower it to flag more at-risk students")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--no-show", action="store_true", help="Don't open the plot window")
    args = p.parse_args()

    df = load_data(args)
    print(df.head(), "\n")
    print(f"Rows: {len(df)} | pass rate: {df[args.target].mean():.1%}\n")

    X = df[args.features].values.astype("float32")
    y = df[args.target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit on train only (no leakage)
    X_test = scaler.transform(X_test)

    # Handle imbalanced pass/fail data by weighting the classes
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}

    model = build_model(len(args.features), args.lr)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True)],
        verbose=0,
    )
    print(f"Stopped after {len(history.history['loss'])} epochs")

    # Evaluation
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {loss:.4f} | accuracy: {acc:.3f} | AUC: {auc:.3f}")
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob > args.threshold).astype(int)
    print(f"\nConfusion matrix (rows = actual, cols = predicted), "
          f"threshold {args.threshold}:")
    print(confusion_matrix(y_test, y_pred))
    print("\n", classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    # Save everything needed for inference
    model.save("student_performance_model.keras")
    joblib.dump(scaler, "scaler.joblib")
    with open("model_config.json", "w") as f:
        json.dump({"features": args.features, "threshold": args.threshold}, f)
    print("Saved: student_performance_model.keras, scaler.joblib, model_config.json")

    # Sample predictions from the held-out set
    print("\nSample predictions (test set):")
    X_orig = scaler.inverse_transform(X_test)
    for row, prob, actual in list(zip(X_orig, y_prob, y_test))[:5]:
        vals = ", ".join(f"{n}={v:.1f}" for n, v in zip(args.features, row))
        label = "PASS" if prob > args.threshold else "FAIL"
        print(f"  {vals} -> {label} (P(pass)={prob:.2f}, actual={'PASS' if actual else 'FAIL'})")

    plot_results(history, model, scaler, X_test, y_test, args.features, "results.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
