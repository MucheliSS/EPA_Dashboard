# SR TPM Appraisal Dashboard

Simple Streamlit dashboard for reviewing SR assessment data from Excel.

## What It Does

- Upload an EPA Excel file.
- Review individual resident domain scores.
- Compare average scores across residents.
- View qualitative comments and basic sentiment outputs.
- Normalize GM scores by dividing them by 2.
- Exclude `0`, blank, `NA`, and non-numeric scores from averages.
- Show optional month-year trends when valid date or month/year data is available.

## Excel Format

Required sheet:

- `Quantitative`

Optional sheet:

- `Qualitative`

Useful columns:

- `Resident Name`
- Domain scores: `PC`, `MK`, `SBP`, `PBLI`, `Prof`, `ICS`, `Overall`
- Assessor: `Name of Evaluator`, `Assessor`, or `Evaluator`
- Date/month: `Assessment Date`, `Feedback Date`, `Month`, `Year`, or similar
- Comments: `Comments`, `Remarks`, `Feedback`, or similar

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run tpm_dashboard.py
```

Then upload your Excel file in the browser.

