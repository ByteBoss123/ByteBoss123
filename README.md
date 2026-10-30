# Amarnath Reddy Ganta

ML engineer and data scientist. M.S. Data Analytics Engineering, George Mason University (May 2026).
I build models and data pipelines on real data and report what the numbers actually say, including when the
simpler baseline wins.

[Portfolio](https://amarnathganta.com) · [LinkedIn](https://www.linkedin.com/in/amarnath-reddy-ganta)

## Experience

- **Machine Learning Engineer Co-Op, CEC TechConnect** (Feb–Apr 2026): built a geospatial clustering and
  logistic regression pipeline over 1,500+ grant applications and $19.4M in requested funding across 8 DC wards,
  cutting analysis time 40% (about 15 hours a week). Built a real-estate valuation pipeline: a 7-source scraper,
  a random-forest valuation model and image-based condition scoring.
- **Data Science Intern, Smart Mieten Tech** (2023–2024): ETL pipelines into warehouse staging layers that cut data
  refresh time 35%; fraud classification models on 300K+ regulated financial records.
- **Data Analyst Intern, Pantechelearning** (2021–2023): SQL data-quality workflows behind six dashboards serving
  1,000+ learners; simulation scripts that cut experiment-planning time 60%.

## Selected projects

Every number below comes from a results file committed in the linked repo.

| Project | What it shows | Headline result |
|---|---|---|
| [SmartDischarge](https://github.com/ByteBoss123/SmartDischarge) | Which diabetic patients should a care team call after discharge? Calibrated XGBoost on 69,973 real encounters (UCI 130 hospitals) | Test AUROC 0.646 (95% CI 0.630–0.661); calling the top 10% catches 2.3x more 30-day readmissions than random; AUROC checked by race, gender and age |
| [News-Feed A/B Test](https://github.com/ByteBoss123/A-B-Testing-Framework-for-News-Recommendation-Systems) | Should we ship a personalized news ranker? A/B workflow on real MSN logs (73,152 impressions, 50,000 readers) | Primary metric not significant (+0.6%, p = 0.56) and the test was 23% powered; new readers lose 7.4%; recommends a hybrid and a ~128k-readers-per-arm test |
| [CardRoute](https://github.com/ByteBoss123/AI-Powered-Customer-Servicing-Pipeline-AWS-ECS-Fargate-) | Route 13,083 real bank customer messages (Banking77) to 6 teams with a fraud safety net | 97.4% team routing; 315 of 320 fraud messages escalated; 88.5% auto-routed at 98.0% accuracy |
| [ComplaintTriage](https://github.com/ByteBoss123/Bank-Complaint-Escalation-Risk-Predictor) | Which bank complaints will end in monetary relief? 184,902 real CFPB complaints, split by time | Test AUROC 0.747 on 2025-26 complaints (structured fields alone 0.715); top 10% of scores find 33% of relief cases; documents drift and banks where the model fails |
| [SupportQA](https://github.com/ByteBoss123/Gen-AI-QA-System) | Suggest expert answers to insurance customer questions from a 27,413-answer library (InsuranceQA), with confidence routing | Correct answer ranked first 28.1% vs 21.1% for BM25, in the top 5 for 46.8%; trained dual encoder + LightGBM reranker |
| [Network Incident Troubleshooting Assistant](https://github.com/ByteBoss123/Network-Incident-Troubleshooting-Assistant) | Log anomaly detection, Neo4j topology graph, LangChain assistant on Amazon Bedrock | Anomaly F1 0.760 on held-out HDFS logs; Llama 3.1 8B answered 94% of 50 evaluation questions with 0 hallucinated ids |
| [FinDocQA](https://github.com/ByteBoss123/Financial-Document-RAG-Evaluation-System) | RAG over SEC filings with automatic failure-mode detectors, deployed on AWS Lambda | Detector false alarms 96% to 3% on 1,052 held-out answers while catching 98.4% of injected wrong figures |
| [Ad Incrementality Warehouse](https://github.com/ByteBoss123/ad-incrementality-dbt) | dbt + Snowflake + Sigma, randomized holdout on 821,737 Criteo users | Ads caused +26.6% visits (stable across samples; the conversion lift varies too much between samples to quote one number); caught a Snowflake precision bug inflating z-scores ~3% |
| [Subprime Auto Loan Charge-Off Risk](https://github.com/ByteBoss123/Subprime-Auto-Loan-Charge-Off-Risk) | Credit risk on 72,665 real loans from SEC ABS filings | Out-of-time AUC 0.656 vs 0.550 for bureau score; Cox survival model cut calibration error 55% |
| [FailSight](https://github.com/ByteBoss123/FailSight) | Remaining-useful-life prediction on NASA C-MAPSS with drift monitoring | LSTM RMSE 12.8 cycles vs 41.9 baseline; flagged 21 of 25 imminent failures with 1 false alarm |
| [Olist Seller KPI Warehouse](https://github.com/ByteBoss123/olist-marketplace-trends) | Star schema on 99,441 real orders, deployed to Redshift | 13/13 integrity tests; late orders average a 2.54 review vs 4.28 on time |
| [Email Campaign Lift](https://github.com/ByteBoss123/email-campaign-lift) | Causal analysis of a 64,000-customer randomized email test in R | +86.5% conversion (p = 4.27e-10), $25,480 incremental revenue (95% CI $16,119–$34,782) |
| [NYC Flight Delays in R](https://github.com/ByteBoss123/NYC-Flight-Delay-Statistics-R) | Hypothesis tests and logistic regression on 336,776 flights | Out-of-time AUC 0.681; riskiest decile delayed 6.3x as often as the safest |
| [CreditWatch Lakehouse](https://github.com/ByteBoss123/Credit-Repayment-Risk-Lakehouse) | PySpark + Delta Lake pipeline executed on Databricks | 30,000 accounts and 180,000 account-months; replay, grain and referential checks pass on Databricks |
| [CausalBench](https://github.com/ByteBoss123/RetailRetain) | Can you evaluate a program with no control group? Naive, DiD, matching, regression, IPW and doubly robust estimates scored against a known randomized effect (NSW job training) | Naive estimate has the wrong sign; doubly robust lands within $156 of the true $1,794 and is the only adjusted estimate whose CI excludes zero; matching swings $376–$2,194 across 20 specifications |
| [GNN vs Fingerprints](https://github.com/ByteBoss123/GNN-MOlecular) | GCN/GIN vs random forest and XGBoost on MoleculeNet, scaffold split, 3 seeds | GCN cuts solubility RMSE 26% (1.01 vs 1.37); on Tox21 toxicity it ties the random forest (0.755 vs 0.752 AUC) |
| [Telco Churn](https://github.com/ByteBoss123/Churn-Analysis) | Churn drivers and call-list ranking on 7,032 customers | Test AUC 0.836; riskiest 20% of customers hold 49.7% of churners |
| [PriceCurve](projects/pricecurve) | Hierarchical Bayesian elasticity (PyMC) + promo optimizer (PuLP) on real retail data | Wrong-sign elasticities 13.7% to 1.0%; targeted promos +130% incremental revenue vs best uniform discount |
| [RingGuard](projects/ringguard) | Graph-based cold-start fraud detection (synthetic data, disclosed) | 20/20 flagged device rings were real rings; behavioral-only model is blind on new accounts (AUC 0.500) |

## Tools I've used in these projects

Python, SQL, R, SAS · scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch, PyMC · PySpark, dbt, Airflow, DuckDB ·
Snowflake, Redshift, Databricks / Delta Lake, Neo4j · AWS (Lambda, Bedrock, S3, EC2) · FastAPI, LangChain
