# Climate Polarisation = f(Stance, Sarcasm, Toxicity)


**fully reproducible mini‑project** with automated content analysis and deliver theory‑linked polarisation metrics from social‑media text.


## Research Questions
- **Q1**: Is sarcasm more prevalent in *against* vs *favor* climate tweets?
- **Q2**: Does **toxicity** predict stance **extremity** (non‑neutral)?
- **Q3**: How polarised is the stance distribution (ER‑style index with CIs)?


## Pipeline
1. **Python** loads TweetEval `stance_climate`, infers **sarcasm** (CardiffNLP) and **toxicity** (Detoxify), saves tidy CSV.
2. **R** computes summaries, runs (multi)logit models, and estimates an **ER‑style polarisation** proxy.


## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python py/01_infer_irony_toxicity.py --out data/tweeteval_climate_with_scores.csv
R -q -e "source('R/02_analyse_polarisation.R')"
```
## Notes

* Labels: 0=none, 1=against, 2=favor.

* is_extreme = 1 if label ∈ {against, favor}.

* stance_num encodes stance as −1/0/+1 for ER proxy.

## Citations (add to your final PDF)

* TweetEval benchmark & stance_climate subset

* CardiffNLP twitter-roberta-base-irony

* Detoxify toxicity models

* Esteban & Ray (1994); Duclos, Esteban & Ray (2004) for polarisation

---


### Optional extensions 
- Run sensitivity with/without `none` class.
- Try regularised models (L2) and interactions `toxicity × sarcasm`.
- Swap in larger climate datasets or add sentiment as a covariate.
