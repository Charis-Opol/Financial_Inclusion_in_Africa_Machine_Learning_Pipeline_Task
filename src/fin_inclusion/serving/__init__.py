"""Online inference service for the production XGBoost/class-weight model.

Deliberately imports nothing from `fin_inclusion.preprocessing` (pandas /
sklearn): every preprocessing constant the service needs is frozen into
`metadata.json` at training time by `scripts/train_production_model.py`, so
the serving image only needs numpy + xgboost + the web stack.
"""
