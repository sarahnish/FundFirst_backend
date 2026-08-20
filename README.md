# FundFirst backend for Lovable

This package serves the real FundFirst Logistic Regression pipeline through a small FastAPI service. The model was reproduced from the submitted notebook's final training procedure: `StandardScaler` followed by balanced `LogisticRegression`, fitted on the 2014–2023 rows of the 72-row labelled FundFirst dataset.

The submitted notebook was inspected only and was not changed.

## Package contents

```text
FundFirst_backend/
├── main.py
├── requirements.txt
├── render.yaml
├── README.md
└── models/
    ├── fundfirst_logistic_regression.joblib
    └── fundfirst_model_metadata.json
```

`main.py` loads the fitted pipeline once at startup and provides `/health`, `/metadata`, and `/predict`. The `models` folder contains the fitted scaler and classifier together in one joblib pipeline, plus the metadata written according to the notebook.

## Run locally

Python 3.12 is recommended.

```bash
cd FundFirst_backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs`, expand `POST /predict`, choose **Try it out**, and submit:

```json
{
  "AveragePrice": 515000,
  "MedianAnnualPay": 35500,
  "SavingRatio": 9.0,
  "BaseRate": 4.25
}
```

Or test from a terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"AveragePrice":515000,"MedianAnnualPay":35500,"SavingRatio":9.0,"BaseRate":4.25}'
```

The response contains the model's class and class probabilities. Raw values must be sent exactly as shown; do not scale them in Lovable because scaling is already inside the saved pipeline.

## Deploy on Render

1. Extract this ZIP and upload the `FundFirst_backend` folder to a new GitHub repository.
2. In Render, select **New > Blueprint** and connect that repository. Render reads `render.yaml` automatically.
3. Deploy the service and wait for `/health` to report `{"status":"ok", ...}`.
4. In Render, replace the `ALLOWED_ORIGINS` value `*` with the exact deployed Lovable origin, for example `https://your-project.lovable.app`. Multiple origins can be comma-separated.
5. Copy the Render service URL, for example `https://fundfirst-api.onrender.com`.

For a manual Render web service, use:

- Runtime: **Python 3**
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

## Connect Lovable

Set Lovable's `VITE_API_URL` environment variable to the deployed backend URL without a trailing slash. On form submission, send a POST request to `${VITE_API_URL}/predict` with:

```json
{
  "AveragePrice": 515000,
  "MedianAnnualPay": 35500,
  "SavingRatio": 9.0,
  "BaseRate": 4.25
}
```

Use `prediction` and `probabilities` from the response as the only prediction source. Do not recreate the model, its scaling, or the TSM thresholds in the frontend.

## Important interpretation

The pipeline reproduces the project's rule-generated deposit-feasibility labels. It does not determine mortgage eligibility, creditworthiness, or mortgage approval and does not provide regulated financial advice.
