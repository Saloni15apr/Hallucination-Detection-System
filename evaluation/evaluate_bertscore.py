from datasets import load_dataset
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
import time


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "potsawee/wiki_bio_gpt3_hallucination"
)["evaluation"]

print(f"Loaded {len(dataset)} passages.")


# --------------------------------------------------
# 2. LOAD SELFCHECK BERTSCORE
# --------------------------------------------------

print("Loading SelfCheckBERTScore...")

selfcheck = SelfCheckBERTScore(
    default_model="en",
    rescale_with_baseline=True
)

print("SelfCheckBERTScore loaded successfully.")


# --------------------------------------------------
# 3. EXPERIMENT SETTINGS
# --------------------------------------------------

N_VALUES = [1, 3, 5, 10, 20]

results = []


# --------------------------------------------------
# 4. RUN EXPERIMENTS
# --------------------------------------------------

for N in N_VALUES:

    print()
    print("=" * 60)
    print(f"N = {N}")
    print("=" * 60)

    all_scores = []
    all_binary_labels = []
    all_severity_labels = []

    total_time = 0.0

    for i, item in enumerate(dataset):

        sentences = item["gpt3_sentences"]
        annotations = item["annotation"]
        samples = item["gpt3_text_samples"][:N]

        start_time = time.perf_counter()

        scores = selfcheck.predict(
            sentences,
            sampled_passages=samples
        )

        elapsed = time.perf_counter() - start_time

        total_time += elapsed

        scores = np.asarray(scores, dtype=float)

        # --------------------------------------------------
        # Convert human annotations to numerical labels
        # --------------------------------------------------

        for score, label in zip(scores, annotations):

            if label == "accurate":

                binary_label = 0
                severity_label = 0.0

            elif label == "minor_inaccurate":

                binary_label = 1
                severity_label = 0.5

            elif label == "major_inaccurate":

                binary_label = 1
                severity_label = 1.0

            else:

                continue

            all_scores.append(float(score))
            all_binary_labels.append(binary_label)
            all_severity_labels.append(severity_label)

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (i + 1) % 10 == 0:
            print(
                f"Processed {i + 1}/{len(dataset)} passages"
            )

    # --------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------

    all_scores = np.asarray(all_scores)

    all_binary_labels = np.asarray(
        all_binary_labels
    )

    all_severity_labels = np.asarray(
        all_severity_labels
    )


    # --------------------------------------------------
    # AUC-PR
    # --------------------------------------------------

    auc_pr = average_precision_score(
        all_binary_labels,
        all_scores
    )


    # --------------------------------------------------
    # Pearson correlation
    # --------------------------------------------------

    pearson = pearsonr(
        all_scores,
        all_severity_labels
    ).statistic


    # --------------------------------------------------
    # Spearman correlation
    # --------------------------------------------------

    spearman = spearmanr(
        all_scores,
        all_severity_labels
    ).statistic


    # --------------------------------------------------
    # Average processing time
    # --------------------------------------------------

    average_time = (
        total_time / len(dataset)
    )


    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print()
    print(f"N = {N}")
    print(f"AUC-PR       : {auc_pr:.4f}")
    print(f"Pearson      : {pearson:.4f}")
    print(f"Spearman     : {spearman:.4f}")
    print(
        f"Time/passage : "
        f"{average_time:.4f} seconds"
    )


    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results.append({

        "N": N,

        "AUC_PR": auc_pr,

        "Pearson": pearson,

        "Spearman": spearman,

        "Time_per_passage_seconds":
            average_time

    })


# --------------------------------------------------
# 5. CREATE RESULTS TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)


print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

results_df.to_csv(
    "table_vi_results.csv",
    index=False
)

print()
print("Results saved to:")
print(
    "evaluation/table_vi_results.csv"
)