# Amarnath Reddy Ganta

Data scientist and ML engineer. M.S. Data Analytics Engineering, George Mason University (May 2026).

I work in two areas:

- **Credit, fraud and financial risk:** models that rank risk, stay calibrated over time and hold up to review.
- **LLM systems and their evaluation:** RAG and agent pipelines, with tests that show when the model is wrong.

Every project uses real data, and every project number below comes from a results file committed in the linked repo,
including the cases where the simpler baseline wins.

[Portfolio](https://amarnathganta.com) · [LinkedIn](https://www.linkedin.com/in/amarnath-reddy-ganta)

## Experience

- **Machine Learning Engineer Co-Op, CEC TechConnect DC** (Feb–Apr 2026): built a geospatial clustering and
  logistic regression pipeline over 1,500+ grant applications and $19.4M in requested funding across 8 DC wards,
  cutting analysis time 40% (about 15 hours a week). Built a real-estate valuation pipeline: a 7-source scraper,
  a random-forest valuation model and image-based condition scoring.
- **Data Science Intern, Smart Mieten Tech** (2023–2024): fraud classification models on 300K+ regulated financial
  records for banking and insurance clients; ETL pipelines into warehouse staging layers that cut data refresh time 35%.
- **Data Analyst Intern, Pantechelearning** (2021–2023): SQL data-quality workflows behind six dashboards serving
  1,000+ learners.

## Credit, fraud and financial risk

| Project | What it shows | Headline result |
|---|---|---|
| [Subprime Auto Loan Charge-Off Risk](https://github.com/ByteBoss123/Subprime-Auto-Loan-Charge-Off-Risk) | Charge-off risk on 72,665 real loans from SEC ABS filings, in Python and SAS | Out-of-time AUC 0.656 vs 0.550 for bureau score; input PSI stayed below 0.012 while the charge-off rate rose 56%, and only a calibration backtest caught it |
| [CreditDesert](https://github.com/ByteBoss123/CreditDesert-Fairness-Gated-Credit-Risk-Modeling-Adverse-Impact-Auditing-System) | HMDA mortgage approval model and consumer credit model with an MLflow registry, group disparity reports, SHAP/PSI monitoring and a FastAPI demo | Test AUC 0.883 on 49,405 held-out HMDA applications; consumer model test AUC 0.864 |
| [ComplaintTriage](https://github.com/ByteBoss123/Bank-Complaint-Escalation-Risk-Predictor) | Which bank complaints will end in monetary relief? 184,902 real CFPB complaints, split by time | Test AUROC 0.747 on 2025-26 complaints (structured fields alone 0.715); top 10% of scores find 33% of relief cases |
| [CreditWatch Lakehouse](https://github.com/ByteBoss123/Credit-Repayment-Risk-Lakehouse) | PySpark + Delta Lake repayment-risk pipeline executed on Databricks | 30,000 accounts and 180,000 account-months; replay, grain and referential checks pass on Databricks |

## LLM systems and evaluation

| Project | What it shows | Headline result |
|---|---|---|
| [FinDocQA](https://github.com/ByteBoss123/Financial-Document-RAG-Evaluation-System) | RAG over SEC filings with automatic failure-mode detectors, deployed on AWS Lambda | Detector false alarms 96% to 3% on 1,052 held-out answers while catching 98.4% of injected wrong figures |
| [CardRoute](https://github.com/ByteBoss123/AI-Powered-Customer-Servicing-Pipeline-AWS-ECS-Fargate-) | Route 13,083 real bank customer messages (Banking77) to 6 teams with a fraud safety net | 97.4% team routing; 315 of 320 fraud messages escalated; 88.5% auto-routed at 98.0% accuracy |
| [Network Incident Troubleshooting Assistant](https://github.com/ByteBoss123/Network-Incident-Troubleshooting-Assistant) | Log anomaly detection, Neo4j topology graph, LangChain assistant on Amazon Bedrock | Anomaly F1 0.760 on held-out HDFS logs; Llama 3.1 8B answered 94% of 50 evaluation questions with 0 hallucinated ids |
| [SupportQA](https://github.com/ByteBoss123/Gen-AI-QA-System) | Suggest expert answers to insurance questions from a 27,413-answer library (InsuranceQA), with confidence routing | Correct answer ranked first 28.1% vs 21.1% for BM25; trained dual encoder + LightGBM reranker |

## Also built

- [Generic drug pricing optimization](https://github.com/ByteBoss123/mac-pricing-optimizer): Gurobi MIP over 297 generics ($12.7B Medicare Part D) meets a 20% savings guarantee by re-pricing 46; CI/CD to Cloud Run.
- [Email Campaign Lift](https://github.com/ByteBoss123/email-campaign-lift): 64,000-customer randomized test in R; +86.5% conversion (p = 4.27e-10), $25,480 incremental revenue.
- [Ad Incrementality Warehouse](https://github.com/ByteBoss123/ad-incrementality-dbt): dbt + Snowflake holdout analysis on 821,737 Criteo users; ads caused +26.6% visits.
- [Olist Seller KPI Warehouse](https://github.com/ByteBoss123/olist-marketplace-trends): star schema on 99,441 real orders, deployed to Redshift; 13/13 integrity tests.

## Tools

Python, SQL, R, SAS · scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow · LangChain, Amazon Bedrock, MLflow ·
PySpark, dbt, Airflow, Databricks / Delta Lake · Snowflake, Redshift, PostgreSQL · AWS, GCP · FastAPI, Docker
