# FedFraud Pipeline v5 — Technical Documentation
### Privacy-Preserving Cross-Bank Fraud Detection (IEEE-CIS Dataset)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Overview](#2-dataset-overview)
3. [Cell 1 — Imports](#3-cell-1--imports)
4. [Cell 2 — Load Data](#4-cell-2--load-data)
5. [Cell 3 — Memory Optimisation](#5-cell-3--memory-optimisation)
6. [Cell 4 — Merge Datasets](#6-cell-4--merge-datasets)
7. [Cell 5 — Data Understanding](#7-cell-5--data-understanding)
8. [Cell 6 — Cleaning](#8-cell-6--cleaning)
9. [Cell 7 — Time Features](#9-cell-7--time-features)
10. [Cell 8 — User ID (UID) Construction](#10-cell-8--user-id-uid-construction)
11. [Cell 9 — Core Behavioural Features](#11-cell-9--core-behavioural-features)
12. [Cell 10 — Simple but Effective Features](#12-cell-10--simple-but-effective-features)
13. [Cell 11 — Encoding Deferral](#13-cell-11--encoding-deferral)
14. [Cell 12 — Final Cleanup](#14-cell-12--final-cleanup)
15. [Cell 13 — Bank Split (Non-IID)](#15-cell-13--bank-split-non-iid)
16. [Cell 14 — Per-Bank Processing & Save](#16-cell-14--per-bank-processing--save)
17. [Cumulative Fixes Summary](#17-cumulative-fixes-summary)
18. [FL Safety Guarantees](#18-fl-safety-guarantees)

---

## 1. Project Overview

FedFraud is a **federated learning (FL) fraud detection pipeline** built on top of the IEEE-CIS Fraud Detection dataset. The core motivation is to simulate a real-world scenario in which multiple banks collaborate to detect fraud without any institution ever sharing raw transaction data with another.

In traditional machine learning, all data is pooled into a central server for training. This is impractical in banking due to regulatory constraints (GDPR, PCI-DSS), competitive sensitivity, and customer privacy obligations. Federated learning resolves this by keeping each bank's data local — only model gradients or weight updates are exchanged between clients and a central aggregation server (e.g., using the FedAvg algorithm).

This pipeline is responsible for the **pre-training data preparation phase**: transforming raw CSVs into clean, FL-safe, per-bank datasets that can be fed directly into Flower FL clients. Every design decision in this pipeline is evaluated against two criteria:

- **Predictive validity** — does this improve fraud detection accuracy?
- **FL safety** — does this step respect the constraint that no bank can see another bank's data, and no future data can influence past predictions?

---

## 2. Dataset Overview

| File | Rows | Columns | Description |
|---|---|---|---|
| `train_transaction.csv` | ~590,540 | 394 | Core transaction records |
| `train_identity.csv` | ~144,233 | 41 | Device and identity metadata |

The two files share a `TransactionID` key. Only ~24% of transactions have a corresponding identity record, making the identity file a sparse supplement rather than a primary source.

The target variable is `isFraud` (binary), with a baseline fraud rate of approximately **3.5%** — a heavily imbalanced classification problem.

---

## 3. Cell 1 — Imports

### What it does
Loads all required Python libraries in a single, centralised block at the top of the notebook.

### Why it matters for the FL project

Centralising imports serves two purposes in a research pipeline like FedFraud:

**Reproducibility**: Anyone picking up this notebook — including FL client developers who may run the preprocessing step on their own machines — can immediately see every dependency in one place. There are no hidden imports buried inside functions.

**Version auditability**: In federated learning, all participating banks must process their data identically. If different banks ran different versions of preprocessing libraries, features would diverge silently. Keeping all imports visible makes it easy to pin versions in a `requirements.txt`.

**Notable v5 change**: `LabelEncoder` was removed because global label encoding was identified as a data leakage source (each bank should not know the global category vocabulary). `RobustScaler` was added to support the new per-bank feature scaling step introduced in Fix v5-8.

---

## 4. Cell 2 — Load Data

### What it does
Reads both raw CSVs from disk and performs basic structural assertions:
- Verifies `TransactionID` exists in both files
- Verifies that the `isFraud` target column contains no nulls
- Prints the baseline fraud rate (~3.5%)

### Why it matters for the FL project

**Structural validation at ingestion** is critical because downstream steps — including the bank split in Cell 13 and the per-bank processing in Cell 14 — assume specific column layouts. A corrupted or truncated CSV would propagate silently into per-bank files without these early guards.

The fraud rate check (`~0.0350`) confirms the dataset is the correct version and has not been accidentally filtered. Because the FL simulation assigns transaction subsets to different banks, preserving the overall fraud rate at ingestion is the baseline against which per-bank fraud heterogeneity is measured.

The `INPUT_PATH` constant is intentionally parameterised (with a local path option commented out) so that FL researchers can point the pipeline at their own data directories without modifying logic code.

---

## 5. Cell 3 — Memory Optimisation

### What it does
Downcasts all numeric columns to their smallest safe dtype:
- Integers: `int64` → `int8`, `int16`, or `int32` (whichever fits)
- Floats: `float64` → `float32` (minimum; `float16` is explicitly avoided)

### Why it matters for the FL project

The merged transaction + identity dataframe can exceed **4 GB** in default `float64` representation. On Kaggle kernels (16 GB RAM) and on bank-side FL clients (which may have constrained hardware), this causes out-of-memory crashes before any actual ML work begins.

The `float32` floor is deliberate. `float16` offers only ~3 significant decimal digits — insufficient for transaction dollar amounts (e.g., $1,234.56 rounds to $1,235 in float16). `float32` provides ~7 significant digits, which is adequate.

Downcasting is applied **before the merge** (on each file independently) and also implicitly afterwards when new derived columns are cast. The typical 40–60% RAM saving enables the full pipeline to run within Kaggle's free tier, making the project accessible to collaborators without GPU/TPU resources.

---

## 6. Cell 4 — Merge Datasets

### What it does
Performs a **left join** of identity data onto transactions using `TransactionID` as the key.

```
df = df_trans.merge(df_id, on="TransactionID", how="left")
```

After the merge, raw frames are deleted and garbage-collected.

### Why it matters for the FL project

**Left join is the correct choice** because it preserves all 590k transactions. An inner join would silently drop the ~76% of card-present transactions with no identity record, skewing the dataset heavily toward online/CNP (card-not-present) transactions and destroying the fraud rate.

The assertion `len(df) == len(df_trans)` explicitly guards against accidental fan-out (duplicates in `df_id` inflating row count). In a federated setting, if some banks received inflated datasets and others did not, the FedAvg weight aggregation would be numerically incorrect because each client's gradient contribution is weighted by dataset size.

Deleting `df_trans` and `df_id` immediately after the merge is standard FL pipeline hygiene — holding intermediate frames in memory while processing per-bank splits doubles peak RAM usage unnecessarily.

---

## 7. Cell 5 — Data Understanding

### What it does
Computes a quick missingness profile and prints:
- Count of columns with >80%, >50%, and 0% missing values
- `card4` value distribution (card network: Visa, Mastercard, Amex, Discover)
- Fraud rate broken down by card network

### Why it matters for the FL project

This is a **diagnostic-only cell** — it writes no new columns and modifies nothing. Its role in the FL pipeline is documentation: it establishes shared ground truth about the dataset that all FL participant banks and the central server team should agree on before deployment.

The card network breakdown is particularly relevant because it reveals natural data heterogeneity. Visa transactions vastly outnumber Discover transactions, and fraud rates differ across networks. These natural patterns justify the Non-IID bank split in Cell 13 and provide a baseline for validating that the split preserves realistic heterogeneity.

---

## 8. Cell 6 — Cleaning

### What it does
Six sub-steps, each addressing a distinct data quality concern:

#### 6a — Drop high-missingness columns
Columns with more than **80% missing values** are dropped. The threshold is set to 80% rather than a lower value because many V-columns (engineered by Vesta, the dataset provider) have structured missingness correlated with transaction type — dropping them at 50% would remove genuinely predictive features.

**FL rationale**: Columns that are 80%+ missing provide almost no signal and force every FL client to impute almost entirely from their training median, meaning the "feature" becomes nearly a constant that varies slightly by bank. Such pseudo-features add noise to FedAvg gradient aggregation without contributing to shared model learning.

#### 6b — Drop constant columns
Columns with ≤1 unique value (including NaN) are removed. Constant features have zero variance and therefore zero gradient contribution during training — they waste bandwidth in FL gradient transmission.

#### 6c — Log transform of TransactionAmt (Fix v5-4)
`TransactionAmt_log = log1p(TransactionAmt)` replaces the previous 99.9th-percentile capping approach.

**Why this fix was critical**: Capping truncates high-value transactions to the cap value. High-value transactions are disproportionately fraudulent in card-not-present scenarios (e.g., large wire initiations, luxury goods purchases on stolen cards). Capping destroys exactly the signal that separates fraud from normal in the tail. `log1p` compresses the scale continuously without destroying any observations — a $50,000 transaction is still distinct from a $5,000 transaction after the transform.

**FL rationale**: The log transform is applied globally before the bank split, so all banks receive identically transformed amounts. The original `TransactionAmt` column is retained alongside for interpretability and for the Card BIN aggregation in Cell 10.

#### 6d — M-flag imputation with "Miss"
The M1–M9 columns are binary match flags (T/F) for various metadata fields. Their NaN state is semantically meaningful — it indicates the metadata field was absent, not that the match result is unknown. Filling with "Miss" preserves this third state and prevents it from being conflated with either True or False.

**FL rationale**: "Miss" is a fixed sentinel — it is not a statistic derived from any data split. This makes it safe to apply globally before the bank split without introducing leakage.

#### 6e — Email domain cleaning (Fix v5-6)
Email domains are normalised via alias grouping (e.g., `gmail.com`, `gmail` → `"Google"`; various Hotmail/Outlook/Live variants → `"Microsoft"`). The top 50 domains by frequency are kept as named categories; the rest are prefixed with `"Rare_"`.

**Why this matters**: The original approach collapsed all rare domains into a single `"Other"` bucket. This destroys identity for rare-but-coherent domain groups (e.g., corporate email domains that may have distinct fraud profiles). Prefixing rare domains with `"Rare_"` preserves their individual identity while signalling to the frequency encoder in Cell 14 that they are low-frequency — they will naturally receive low frequency encoding values.

#### 6f — Categorical NaN → "UNKNOWN"
All remaining object-dtype columns have NaN replaced with the string `"UNKNOWN"`.

**Critical distinction from Fix v5-1**: This is safe to do globally (before the bank split) because "UNKNOWN" is a fixed sentinel, not a statistic. No data leakage occurs. Numeric imputation, which *is* a statistic (the median), has been deliberately removed from this cell and moved to Cell 14 (per-bank, train-only).

---

## 9. Cell 7 — Time Features

### What it does
Converts `TransactionDT` (seconds since a reference epoch of 2017-11-30) into interpretable datetime components:

| Feature | Description |
|---|---|
| `_Hour` | Hour of day (0–23) |
| `_DayOfWeek` | Day of week (0=Mon, 6=Sun) |
| `_Day` | Calendar day of month |
| `_Month` | Calendar month |
| `_is_late_night` | Binary: 1 if hour is 0–5, else 0 |

After feature creation, **the dataframe is sorted by `TransactionDT`**. This sort is not optional.

### Why it matters for the FL project

**The sort is mandatory for correctness of Cells 8–10.** All behavioural aggregations (tx_count_uid, time_diff_uid, amt_z_uid) use `cumsum` and `cumcount` to compute past-only statistics. These operations are order-dependent: the "past" for each row is defined by its position in the sorted sequence. Running them on an unsorted dataframe would silently include future transactions in past-window aggregates, introducing severe target leakage.

The `_is_late_night` flag is included because fraudulent card-not-present transactions skew heavily toward late-night hours (when the legitimate cardholder is less likely to be monitoring their account). This is a low-computation, high-signal binary feature that adds meaningful discriminative power for all four bank clients.

`TransactionDT` itself is **not dropped here** — it is still needed in Cell 13 for the bank split clustering (median TransactionDT per card1) and in Cell 9 for `time_diff_uid` computation. It is dropped in Cell 14, just before saving.

---

## 10. Cell 8 — User ID (UID) Construction

### What it does
Constructs a synthetic per-user identifier by concatenating five fields:

```
uid = card1 * card2 * addr1 * addr2 * P_emaildomain
```

Missing values in any field are replaced with `"UNK"` before concatenation. After the bank split in Cell 13, each UID is made bank-local by appending the bank ID (`uid + "_" + bank_id`).

### Why it matters for the FL project

**A strong UID is the foundation of all behavioural features.** The tx_count, time_diff, and amount z-score aggregations in Cell 9 are all computed per-UID. If UIDs conflate multiple distinct users, those aggregates become noisy averages over heterogeneous behaviour rather than clean per-user histories.

**v4 weakness**: Using only `card1_card2` as the UID conflates users who happen to share the same card BIN prefix and card suffix — a surprisingly common occurrence given that BINs are assigned to banks in ranges, and many legitimate cards share the first 6 digits. Adding billing address (addr1, addr2) and the purchaser's email domain creates a much stronger identity proxy without requiring any PII.

**Bank isolation (Fix v5-3 corollary)**: After the bank split, UID values are scoped to the bank by appending `bank_id`. This prevents behavioural aggregates from one bank's training data leaking into another bank's client model during FL rounds. If a user appears in both bank_A and bank_B (e.g., because they have cards issued by both institutions), their UID-level history within each bank remains independent.

---

## 11. Cell 9 — Core Behavioural Features

### What it does
Computes four past-only behavioural aggregates per UID using vectorised cumulative operations:

| Feature | Description |
|---|---|
| `tx_count_uid` | Number of past transactions for this UID |
| `is_first_txn` | Binary: 1 if this is the user's first transaction |
| `time_diff_uid` | Seconds since the UID's last transaction |
| `amt_z_uid` | Z-score of current amount vs UID's past amounts |
| `amt_ratio_uid` | Ratio of current amount to UID's past mean amount |

### Why each feature was included

**tx_count_uid**: Velocity is one of the most powerful fraud signals. Fraudsters tend to make many rapid transactions immediately after compromising a card. New accounts with high velocity are high-risk; established accounts with consistent low velocity are low-risk.

**is_first_txn (Fix v5-5)**: First transactions have no prior history, making their `time_diff_uid` and `amt_z_uid` undefined. Creating an explicit binary flag allows the model to learn a separate decision rule for first-time users, rather than treating the imputed constant (180 days) as a meaningful time gap.

**time_diff_uid (Fix v5-5)**: Short time gaps between successive transactions signal rapid-fire fraud. The v4 approach filled the first transaction's time gap with the median of all time gaps — semantically wrong, because the first transaction has no prior event and the median of other users' gaps is irrelevant. The fix fills first-transaction gaps with 180 days (a large constant representing "no recent history") and uses `is_first_txn` to let the model distinguish this case.

**amt_z_uid and amt_ratio_uid**: Fraudsters typically attempt high-value transactions relative to the cardholder's historical spending. A purchase that is 5 standard deviations above a user's typical spend is a strong fraud signal regardless of the absolute dollar amount. These features normalise for per-user spending level, making the model robust to the fact that high-income cardholders legitimately spend more in absolute terms.

**Past-only vectorisation**: All aggregates are computed using `cumsum - current_value` and `cumcount`, which are O(n) vectorised operations. Using a Python loop with `.shift()` would be correct but would take hours on 590k rows. The vectorised pattern ensures the pipeline runs in minutes while remaining mathematically identical.

### Why it matters for the FL project

These features are computed **globally before the bank split**, which is correct. They depend only on the temporal ordering of transactions (already established by Cell 7's sort) and on UID assignments (established in Cell 8). The bank split in Cell 13 is a downstream step — it partitions a dataframe that already has clean, leakage-free behavioural features.

---

## 12. Cell 10 — Simple but Effective Features

### What it does
Adds two sets of features:

**Email match flag**:
```
email_match = 1 if P_emaildomain == R_emaildomain AND domain != "UNKNOWN"
```

**Card BIN aggregations (past-only)**:
- `card1_tx_count`: number of past transactions on this BIN
- `TransactionAmt_to_card1_mean`: ratio of current amount to the BIN's historical mean

### Why each feature was included

**email_match**: A mismatch between the purchaser's email domain and the recipient's email domain is a moderate-strength fraud signal. For example, a Visa card registered to a Google email address making a payment to a Microsoft email address may indicate a compromised account. More importantly, when both emails are identical and known (not "UNKNOWN"), it confirms account consistency — a legitimate user typically sends to themselves or known contacts.

**card1_tx_count and TransactionAmt_to_card1_mean**: These are BIN-level (not UID-level) features. They capture the transaction activity pattern of the entire issuing bank's card range, not just an individual cardholder. A BIN that suddenly receives a spike in transaction volume (high `card1_tx_count`) or abnormally large amounts relative to its historical mean (high ratio) may indicate a targeted BIN-level attack — a scenario where fraudsters have obtained a batch of card numbers from the same issuing bank.

Both card1 features are computed as **past-only** using the same `cumsum - current_value` pattern as Cell 9. This was identified as a bug in v3 (Fix v4-6) and corrected: using full-dataset `card1` statistics would include future transaction amounts in the "mean", inflating or deflating the ratio in ways that cannot be replicated at inference time.

---

## 13. Cell 11 — Encoding Deferral

### What it does
This cell is a **placeholder and planning cell**. It identifies all object-dtype columns that will need encoding and stores them in `CAT_COLS_FOR_ENCODING`. No transformation is applied.

### Why the encoding was moved (Fix v5-3)

In v4, a global `LabelEncoder` was fitted on the full merged dataset before the bank split. This approach has two FL-specific problems:

**Problem 1 — Data leakage**: Fitting an encoder on the full dataset means each bank's categorical vocabulary is derived from data that includes other banks' transactions. In a real FL deployment, bank_A has no access to bank_B's data — it cannot know that a particular email domain appears in bank_B's transaction history but not its own.

**Problem 2 — Integer encoding breaks FedAvg**: Label encoding assigns arbitrary integers to categories (e.g., "Google" → 3, "Yahoo" → 7). If bank_A's encoder maps "Google" → 3 and bank_B independently encodes → 3 as well, this is a coincidence — without a shared global vocabulary, the same integer could mean completely different categories. When FedAvg averages the model weights, the gradient for "feature 3" from bank_A and "feature 3" from bank_B would be for semantically different inputs, silently corrupting the aggregated model.

**Replacement**: Frequency encoding with global temporal-safe maps (computed in Cell 14's preamble) resolves both problems. The same category string maps to the same float value across all banks because the frequency map is computed once on the earliest 70% of the dataset and distributed.

---

## 14. Cell 12 — Final Cleanup

### What it does
- Detects and replaces any `±inf` values with `NaN`
- Confirms `TransactionID` and `uid` are still present (needed for Cell 13's merge)
- Prints a final shape, total NaN count, and fraud rate sanity check

### Why it matters for the FL project

Infinity values arise from division operations in Cells 9 and 10 (e.g., dividing by a near-zero past mean). Left uncorrected, they propagate silently: XGBoost converts `inf` to `NaN` internally without warning; neural-network FL clients produce `NaN` gradients that corrupt the entire FedAvg round when aggregated.

The cleanup converts `inf → NaN` rather than `inf → 0` or `inf → median`, because the per-bank median imputation in Cell 14 will handle NaN values correctly with the right training-only statistics. Filling with a global constant here would be a form of data leakage (the global dataset's extreme values influence the imputed value).

The NaN count at this stage is large and **intentional** — it reflects the numeric NaNs from the original dataset that were not imputed in Cell 6 (by design, per Fix v5-1). The comment "imputed per-bank" signals to any reader that this is expected.

---

## 15. Cell 13 — Bank Split (Non-IID)

### What it does
Partitions all 590k transactions into four simulated banks using a **card1 BIN-level temporal quantile clustering** approach. Each unique `card1` value is assigned permanently to exactly one bank.

### The bank assignment algorithm

1. For each unique `card1` value, compute its median `TransactionDT` (temporal centre of mass).
2. Sort `card1` values by this median timestamp.
3. Assign to four banks using equal-count quantile buckets (`pd.qcut` with `q=4`).

This produces:
- **Temporal locality**: Each bank's transactions are concentrated in a time window, simulating a scenario where different banks' customers shop at different periods (e.g., holiday vs. back-to-school peaks).
- **Non-IID**: Different BIN ranges have different issuing banks, which have different customer demographics, geographic concentrations, spending patterns, and fraud exposure. This creates realistic heterogeneity.

### Why card-network splitting was wrong (Fix v5-7)

The v4 split used `card4` (Visa/Mastercard/Amex/Discover) to assign rows to banks. This has two critical flaws:

**Flaw 1 — It is not how banks work in reality.** Multiple real banks issue Visa cards. Splitting by card network creates synthetic banks where "bank_visa" includes Chase, Bank of America, Wells Fargo, and thousands of other Visa issuers — an incoherent grouping.

**Flaw 2 — Amex and Discover are severely underrepresented.** At ~3.5% and ~1.8% of transactions respectively, these "banks" would have tiny datasets leading to poor FL convergence and unrepresentative gradient updates in FedAvg.

### Non-IID validation

The cell computes a validation table reporting fraud rate, average transaction amount, card network mix, and geographic distribution per bank. The key assertion is:

```
fraud_spread = max(fraud_rate) - min(fraud_rate) > 0.005
```

A spread below 0.5 percentage points would indicate the split failed to create meaningful heterogeneity — the four banks would be nearly identical, defeating the purpose of the Non-IID FL simulation.

### UID bank isolation

After the split, UIDs are made bank-local:
```python
df["uid"] = df["uid"] + "_" + df["bank_id"]
```

This prevents a scenario where the same user (same UID) appears in two banks' training sets and their behavioural history becomes a cross-bank leakage channel.

---

## 16. Cell 14 — Per-Bank Processing & Save

### What it does
The most complex cell in the pipeline. For each of the four banks, it executes a five-step processing sequence, then saves four output files plus a metadata JSON.

---

### Step 0 — Train / Val / Test Split (Chronological OOT)

Rows are split **by position** in the time-sorted dataframe:

| Split | Proportion | Rows |
|---|---|---|
| Train | 70% | First 70% chronologically |
| Validation | 15% | Next 15% |
| Test | 15% | Final 15% |

**Why chronological (OOT)?** Random splitting would allow future transactions to appear in the training set and past transactions to appear in the test set. This is not how fraud models are evaluated in production — a model trained on January data is never evaluated on February data that was randomly intermixed with January. Out-of-time (OOT) splitting correctly simulates the temporal deployment scenario and prevents feature leakage from future rows.

---

### Step 1 — Numeric Imputation, Train-Only Medians (Fix v5-1)

```python
train_medians = X_tr[num_cols].median()
X_tr  = X_tr.fillna(train_medians)
X_val = X_val.fillna(train_medians)
X_te  = X_te.fillna(train_medians)
```

**Why this is the critical data leakage fix**: Computing medians on the full dataset (as in v4) means the imputed value for a training row is influenced by validation and test rows. In a federated setting, this is a double violation: future data leaks into the past, and one bank's data statistics influence another bank's imputation. Per-bank, train-only medians ensure that each client's model is trained on statistics it could legitimately compute from its own historical data.

Medians are stored in `metadata.json` so that deployed FL inference clients can impute incoming real-time transactions using the same values the model was trained with.

`.fillna(0)` is applied as a second-pass fallback for any column where the training split is entirely NaN — an edge case that can occur for sparse identity columns in small banks.

---

### Step 2 — Frequency Encoding with Global Temporal-Safe Maps (Fix v5-3, v5-6)

```python
# Global frequency map computed on earliest 70% of dataset (pre-split)
freq = df_global_train[col].value_counts(normalize=True).to_dict()

# Applied per-bank
X_tr[col]  = X_tr[col].map(freq).fillna(0.0)
X_val[col] = X_val[col].map(freq).fillna(0.0)
X_te[col]  = X_te[col].map(freq).fillna(0.0)
```

**Why global maps, not per-bank maps?** This is a deliberate architectural decision for FL compatibility. FedAvg requires that feature `i` means the same thing in every client's model. If bank_A's frequency encoder maps `"Google" → 0.42` (because Google email is common among its customers) and bank_B maps `"Google" → 0.17`, then the gradient for "feature: email_domain" cannot be meaningfully averaged across clients — the feature represents different semantics.

**Why temporal-safe?** The global map is computed on the earliest 70% of transactions sorted by time. This ensures the frequency map is derived from "past" data and does not incorporate "future" frequencies that would not be available at training time in a real deployment.

**Unseen categories** (those not in the global map) receive frequency 0.0. This is correct: an entirely new email domain at inference time should be treated as having negligible historical prevalence.

---

### Step 3 — Feature Scaling with RobustScaler (Fix v5-8)

```python
scaler = RobustScaler()
X_tr[scale_cols]  = scaler.fit_transform(X_tr[scale_cols])
X_val[scale_cols] = scaler.transform(X_val[scale_cols])
X_te[scale_cols]  = scaler.transform(X_te[scale_cols])
```

**Why RobustScaler?** `StandardScaler` uses mean and standard deviation, which are sensitive to outliers. Transaction data contains extreme outliers (e.g., $500,000 wire transfers). A single outlier can shift the mean and inflate the standard deviation, compressing the bulk of the distribution into a tiny range. `RobustScaler` uses the **median and IQR (interquartile range)** — statistics that are inherently resistant to outliers.

**Why train-only fitting?** Fitting the scaler on validation or test data would incorporate future statistical information into the transform applied to the training set (indirect leakage). The scaler center (median) and scale (IQR) are stored in `metadata.json` so that FL inference clients apply identical scaling to live transactions.

**Why frequency-encoded columns are excluded from scaling**: Frequency encoding already maps categories to values in [0, 1]. Applying RobustScaler to these would re-scale them away from this bounded range without benefit, and would make the scaler parameters unnecessarily sensitive to the frequency distribution of categorical values.

---

### Step 4 — Validation Checks (Fix v5-9)

Four assertions are run after processing:

| Check | What it catches |
|---|---|
| No NaN in any split | Failed imputation or encoding |
| No object columns remaining | Encoding step missed a column |
| Feature columns match across train/val/test | Column ordering inconsistency |
| Fraud rate drift warning | Concept drift between time periods |

**Why these checks matter in FL**: In a multi-client FL training loop, a single bank's malformed data can corrupt an entire federated round. NaN values produce NaN gradients, which propagate through FedAvg aggregation and can crash all clients simultaneously. Schema mismatches between splits would cause shape errors at inference time. Running these checks before saving guarantees that each bank's output is a valid FL training input.

The fraud rate drift check is a **warning, not an assertion**. Concept drift (fraud patterns changing over time) is real and expected. The model must learn to generalise across this drift — crashing the pipeline when drift is detected would be incorrect. Instead, the warning alerts the FL researcher to investigate whether the drift is severe enough to require special handling (e.g., drift-adaptive FL algorithms).

---

### Step 5 — Save CSVs and Metadata (Fix v5-10)

Each bank's output directory contains:

```
banks/bank_A/
├── transactions.csv   (full bank dataset, all splits combined)
├── train.csv          (70% chronological)
├── val.csv            (15% chronological)
├── test.csv           (15% chronological)
└── metadata.json
```

The `metadata.json` file contains every statistic computed during processing:

```json
{
  "bank_id": "bank_A",
  "n_records": 147635,
  "fraud_rate_train": 0.0341,
  "medians": { "TransactionAmt": 68.5, "_Hour": 14.0, ... },
  "frequency_maps": { "P_emaildomain": { "Google": 0.42, ... } },
  "scaler_params": {
    "center": { "tx_count_uid": 2.0, ... },
    "scale": { "tx_count_uid": 4.0, ... }
  }
}
```

**Why metadata is essential for FL**: In federated learning, the central server orchestrates training but does not process data. When a new FL client is initialised (e.g., a new bank joining the federation), it needs to know exactly how to preprocess its own data to match the feature space the global model was trained on. The metadata.json provides a complete, self-contained specification of every transformation applied to that bank's training data — enabling new clients to replicate the preprocessing without access to the original pipeline code.

---

### Cross-Bank Schema Validation

After all four banks are saved, the pipeline re-reads each bank's CSV header and confirms that all four have **identical column names in identical order**. This is the final guard against subtle schema divergence that could occur if, for example, a bank's training split happened to have no instances of a particular category, causing a column to be dropped.

```python
assert all_bank_schemas[bid] == ref_schema
assert disk_feature_cols == ref_schema  # re-read from disk
```

---

## 17. Cumulative Fixes Summary

### Fixes inherited from v3 and v4

| Fix ID | Description |
|---|---|
| v3-1 | Removed `card1_fraud_mean` — target leakage via full-dataset isFraud mean |
| v3-2 | UID aggregates made past-only using vectorised cumsum pattern |
| v3-3 | Bank split aligned via `TransactionID` merge instead of fragile positional sort |
| v3-4 | `TransactionID` kept until save loop in Cell 14 |
| v4-5 | Column collision: raw card4/addr1/addr2 re-merged with `_raw` suffix |
| v4-6 | Card BIN features (Cell 10) made past-only |
| v4-7 | Email domain `isin` checks applied to lowercased copy |
| v4-8 | `TransactionDT` dropped before save |
| v4-9 | `isFraud` column ordering corrected in split CSVs |

### New fixes introduced in v5

| Fix ID | Description | Cell |
|---|---|---|
| v5-1 | Per-bank, train-only median imputation replaces global imputation | 14 |
| v5-2 | Stronger 5-field UID: card1 × card2 × addr1 × addr2 × P_emaildomain | 8 |
| v5-3 | Global temporal-safe frequency encoding replaces LabelEncoder | 11, 14 |
| v5-4 | Log transform (log1p) replaces 99.9th-percentile amount capping | 6 |
| v5-5 | time_diff_uid: 180-day constant + is_first_txn flag for first transactions | 9 |
| v5-6 | Rare email domains: top-50 kept + frequency encoding (not "Other" collapse) | 6 |
| v5-7 | Card1 BIN-level temporal quantile clustering replaces card-network split | 13 |
| v5-8 | RobustScaler per-bank, train-only; parameters stored in metadata | 14 |
| v5-9 | Validation assertions: NaN, schema, object columns, fraud rate drift | 14 |
| v5-10 | Enhanced metadata.json: medians, frequency_maps, scaler_params | 14 |

---

## 18. FL Safety Guarantees

The following table summarises the FL safety properties guaranteed by the v5 pipeline:

| Property | How it is enforced |
|---|---|
| **No cross-bank data leakage** | All statistics (medians, frequency maps, scaler params) computed on training split only; bank split ensures card1-level isolation |
| **No future leakage** | Chronological OOT split; past-only cumulative aggregates; global frequency map from earliest 70% only |
| **Feature semantic consistency** | Global temporal-safe frequency maps ensure the same category string maps to the same float across all banks |
| **UID isolation** | UIDs scoped to bank by appending bank_id; prevents cross-bank behavioural history sharing |
| **Gradient integrity** | No NaN or object columns in any saved CSV; validated by post-processing assertions |
| **Reproducibility** | All per-bank transform parameters saved in metadata.json; pipeline can be re-run identically |
| **Schema consistency** | Cross-bank schema validated on both in-memory and on-disk representations |
| **Non-IID realism** | Card1 BIN temporal clustering produces meaningful fraud rate heterogeneity (>0.5 pp spread required) |

---

*Document version: v5 | Pipeline: FedFraud | Dataset: IEEE-CIS Fraud Detection*
