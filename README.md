# Credit Default Probability Model

A local, CSV-driven baseline that estimates default probability from labeled historical data. Bring your own data; the repository does not include generated application records or claim a measured score.

## Train and evaluate

Requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
python credit_scoring.py train --data path/to/labeled_applications.csv --target defaulted --id-column application_id --test-size 0.2
```

The target must contain numeric `0` (no default) and `1` (default) labels. Remaining columns are treated as features except for an optional ID column, which is excluded from training. Numeric inputs are median-imputed and scaled; categorical inputs are imputed and one-hot encoded. A logistic regression model is fit on a stratified training split.

Training writes holdout probabilities, a JSON report with ROC AUC, average precision, Brier score and log loss, and a reusable `joblib` artifact. Default output paths are `credit_holdout_predictions.csv`, `credit_metrics.json`, and `credit_model.joblib`. Use `--predictions`, `--metrics`, and `--model` to select other paths; `--seed` changes the reproducible split.

## Score new applications

```bash
python credit_scoring.py predict --model credit_model.joblib --input path/to/new_applications.csv --output credit_probabilities.csv
```

The prediction file contains source row numbers, an optional configured ID, and `estimated_default_probability`. It does not emit approval/decline decisions or convert probabilities into a consumer credit score.

## Docker

```bash
docker build -t credit-default-model .
docker run --rm -v /absolute/path/to/data:/data credit-default-model train --data /data/labeled_applications.csv --target defaulted --id-column application_id --model /data/credit_model.joblib --metrics /data/credit_metrics.json --predictions /data/credit_holdout_predictions.csv
```

Mount a host directory containing the CSV and where you want artifacts written. The application reads and writes local files and makes no network requests.

## Scope

This is a technical baseline, not a lending decision system or a claim of regulatory compliance. A random holdout may overstate performance when records from the same people or time periods appear in both splits. Before any consequential use, qualified reviewers need to assess data quality, calibration, fairness, leakage, and the applicable requirements for the intended setting.

## License

MIT. See [LICENSE](LICENSE).
