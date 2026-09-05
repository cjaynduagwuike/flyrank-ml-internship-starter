# Capstone Report — Ranking Signal Analysis

- **Author:** Jasper Nduagwuike
- **Lane:** Ranking SIgnal Analysis
- **Repo:** [Link Here](https://github.com/cjaynduagwuike/flyrank-ml-internship-starter)
- **Date:** 5th September 2026

## 1. Abstract
Abstract (5-sentence summary: question → method → result) 


Search signals associated with changes in visibility, clicks, engagement or movement are at unidentified in large datasets with enormous volumes of data. This paper aims to justify a signal analysis model that identifies or appears to identify, percieved associations between search signals and the various metrics stated above.The model was trained on FlyRank SEO data spanning the period of Februrary 2026, with the decline label created based on impressions in March 2026. The model made use of a client aware split and a binary, 0 or 1, decline label to rank the association between identifiable search signals in pages and the above stated metrics. The model achieved a Precision@20 of 35% (on predicting the binary decline label) opposed to a base decline rate of 18%. As such the model achieved a fairly acceptable benchmark for a simple training technique (logistic regression)

## 1. Problem framing

What decision does this support? Name the unit of analysis (page, client, day…), the output
(score, rank, cluster, report), the action a human takes from it, and the cost of a wrong
call. Why does data/ML help here at all?

The model's ranked queue aids human page reviewers in allocating resources to pages more likely to benefit positively from a refresh. Rather than basing such choices on a rigid rule, a model that learns the underlying patterns in the data, would be better at creating an actionable human review queue. The grain used in the cleaned cache was 1 unique content item per client over Februrary 2026 (`content_hash_id` x `client_hash_id` over Februrary 2026). The model outputs a ranked queue in order of the page it thinks requires the most urgent refresh based on the information ingested by it during its training. The model will never be 100% correct in its recommendations and as such caution is required by human review teams before accepting the model's recommendations as a wrongly suggested refresh call, constitutes to a waste in a human reviewers manpower and resources.

## 2. Data safety

Which data you used and which columns you deliberately excluded (and why). Leakage risks you
considered — especially label-derived fields (`trend_direction`, `trend_pct`) and pseudonymous
IDs (grouping only, never features). Confirm nothing client-identifying appears anywhere in
`work/`.

Data used in training the model was accessed from the FlyRank HuggingFace warehouse release `fact_content_daily_performance` table which contains 78.8 million daily rows. The daily fact table was aggregated into a February-March features parquet table. Columns such as `prior_impressions`, `prior_ctr`, `prior_avg_position`,`prior_sessions` and `prior_engagement_rate` were used as model features. Colummns such as:
- March 2026 features were not added in training the model because they were what the model was tested on.
- `client_hash_id` and `content_hash_id` as predictive features, because they are pseudonymous identifiers. `client_hash_id` was retained only for grouped splitting and evaluation; `content_hash_id` was retained for identifying output rows. .
- `future_impressions` column: Excluded because it contains information from the target window and would cause data leakage if added to the model training data.
- `future_decline_label` column: Excluded because it is the target and as such can never be a feature

During the data aggregation phase, I took special care to avoid the introduction of post-decision features into my cached dataset

## 3. Baseline

The transparent rule or score you built first. Why it's a fair comparison, and its numbers on
the same data and metric as your model.

## 4. Model / analysis

Your method and why it fits the lane. The exact feature list (and what you left out on
purpose). The target or proxy definition, in one sentence.

## 5. Evaluation

Your split (grouped by client? time-aware?) and why. Metrics, model vs baseline **on the same
split**. What the errors look like — a short error analysis beats a big metric table.

## 6. Interpretation

What the model/clusters actually found. Feature importances or cluster profiles in plain
words. Surprises and negative results — a well-understood "no effect" is a valid result.

## 7. Recommendation

The ranked actions or decisions your output supports, and how a FlyRank editor would use them
tomorrow. State your confidence and the limits explicitly.

## 8. Reproducibility

The exact commands to re-run everything from a fresh clone, your random seeds, and your
environment (`pip freeze` highlights or `requirements.txt` deltas).

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> **Metrics vs. base rate:** report your task's base rate (majority-class %) next to any
> precision@K or accuracy — a high score can just be a high base rate. AUC / lift over
> baseline are the honest discrimination numbers.
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
