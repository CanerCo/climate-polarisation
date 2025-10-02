---
title: "Climate Polarisation from Stance, Sarcasm, and Toxicity"
date: "October 01, 2025"
---

## Motivation
We measure how rhetorical signals on Twitter/X relate to polarisation in the climate debate. Using stance labels (*favor / against / none*), automated **sarcasm** and **toxicity** scores, and simple logit/multinomial models, we ask:

- **Q1**: Is sarcasm more prevalent in *against* than *favor* tweets?
- **Q2**: Does **toxicity** increase the odds of an **extreme** (non‑neutral) stance?
- **Q3**: How polarised is the stance distribution?

## Data & Models
- **Dataset**: TweetEval *stance_climate* (N = 564 tweets). Stance distribution — **favor 335 (59.4%)**, **none 203 (36.0%)**, **against 26 (4.6%)**.
- **Sarcasm**: Twitter RoBERTa irony classifier → probability in [0,1].
- **Toxicity**: Detoxify (unbiased) aggregate toxicity score in [0,1].
- **Extremity**: `is_extreme = 1` if stance ∈ {favor, against} else 0.

## Methods (reproducible)
1. **Python** (`01_infer_irony_toxicity_2.py`) adds sarcasm & toxicity to each tweet → `tweeteval_climate_with_scores.csv`.
2. **R** runs:
   - group means (Q1), 
   - **logit**: `extreme ~ toxicity + sarcasm` (Q2),
   - **multinomial** for {none, against, favor} (Q3),
   - ER‑style polarisation proxy with bootstrap CI (Q3).

## Results

### Q1 — Sarcasm by stance
- Mean **P(sarcasm)**: **against = 0.603** (n = 26), **favor = 0.437** (n = 335).
- Difference (against − favor) = **0.166** with 95% CI **[0.048, 0.283]**.
- Cohen’s d = **0.55** → *moderate* effect.  
**Interpretation:** *Against* tweets are notably more sarcastic.

![Q1: Sarcasm probability by stance](sandbox:/mnt/data/q1_sarcasm_by_stance.png)

### Q2 — Does toxicity drive extremity?
Logit (extreme vs none), controlling for sarcasm:
- **Toxicity** OR = **0.412**, 95% CI **[0.162, 1.036]**, *p* = 0.059.
- **Sarcasm** OR = **1.382**, 95% CI **[0.776, 2.476]**, *p* = 0.274.

**Interpretation:** We find **no clear evidence** that higher toxicity increases extremity; if anything, the point estimate trends **down** (borderline).

![Q2: Toxicity → Extremity effect](sandbox:/mnt/data/q2_toxicity_effect.png)

### Q3 — Stance distribution & polarisation
- Shares: **favor 59.4%**, **none 36.0%**, **against 4.6%**.
- ER‑style polarisation proxy = **0.864**, 95% CI **[0.817, 0.901]**.

![Q3: Stance distribution](sandbox:/mnt/data/q3_stance_distribution.png)

### Multinomial (vs “none”)
- **Against ↔ Sarcasm:** OR = **7.223**, 95% CI **[1.808, 28.857]**, *p* < .01 → strong link.
- **Favor ↔ Toxicity:** OR = **0.391**, 95% CI **[0.151, 1.01]**, *p* ≈ 0.052 → more toxicity → *lower* odds of favor (borderline).

## Robustness & Notes
- **Class imbalance:** very few *against* tweets (n = 26).
- **Domain shift:** sarcasm/toxicity models are general‑purpose.
- **Near‑threshold p‑values** (≈ .05–.06) should be interpreted with caution.

## Reproducibility
```bash
# Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 01_infer_irony_toxicity_2.py --in data/tweeteval_climate.csv --out data/tweeteval_climate_with_scores.csv

# R
R -q -e "source('R/02_analyse_polarisation.R')"
```

---
*Figures:*  
- Q1: `q1_sarcasm_by_stance.png`  
- Q2: `q2_toxicity_effect.png`  
- Q3: `q3_stance_distribution.png`

*Tables:*  
- `q1_sarcasm_group_means.csv`  
- `q2_logit_odds_ratios.csv`  
- `multinomial_or.csv`  
- `polarisation_summary.csv`
