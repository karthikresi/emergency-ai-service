# Final Model Selection

| Target | Baseline model | Best candidate | Final model | Primary metric | Final score |
|---|---|---|---|---|---|
| Category | Logistic Regression | Logistic Regression | Logistic Regression | Macro F1 | 0.79 |
| Severity | Linear SVM | Linear SVM | Linear SVM | Macro F1 | 0.74 |
| Required Skill | One-vs-Rest Logistic Regression | One-vs-Rest Logistic Regression | One-vs-Rest Logistic Regression | Macro F1 | 0.72 |
| Responder | Rule + ranking | Rule + ranking | Rule + ranking | subset accuracy | 0.76 |

## Notes
The synthetic prototype intentionally favors interpretable, lightweight models that are easy to deploy and validate in a FastAPI service.
