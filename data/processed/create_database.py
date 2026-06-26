"""
Creates and populates jobs.db with 50 job records and 20 company records.
Run from the project root: python data/processed/create_database.py
"""

import os
import sys
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "database" / "jobs.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_COMPANIES = """
CREATE TABLE IF NOT EXISTS companies (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    industry      TEXT,
    stage         TEXT,
    size          TEXT,
    tech_stack    TEXT,
    culture_summary TEXT,
    founded_year  INTEGER
);
"""

CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY,
    title            TEXT NOT NULL,
    company          TEXT NOT NULL,
    location         TEXT,
    salary_min       REAL,
    salary_max       REAL,
    experience_min   REAL,
    experience_max   REAL,
    skills_required  TEXT,
    industry         TEXT,
    company_stage    TEXT,
    remote_friendly  INTEGER,
    full_description TEXT
);
"""

# ── Company data (20 records) ─────────────────────────────────────────────────

COMPANIES = [
    # Seed (1–7)
    (1,  "NeuralNest AI",       "Artificial Intelligence", "seed",     "11-50",   "Python, PyTorch, FastAPI",
     "Research-first culture; we ship papers and products simultaneously.", 2022),
    (2,  "DataSprout",          "Data Analytics",          "seed",     "1-10",    "Python, dbt, Metabase",
     "Lean team, big autonomy. Everyone owns a vertical.", 2023),
    (3,  "CloudSeed Labs",      "Cloud Infrastructure",    "seed",     "11-50",   "Go, Kubernetes, Terraform",
     "Founders from Google SRE; reliability is a first-class feature.", 2022),
    (4,  "Stackr",              "Developer Tools",         "seed",     "1-10",    "Node.js, React, PostgreSQL",
     "Build for developers, by developers. No product managers.", 2023),
    (5,  "BioSignal Tech",      "HealthTech",              "seed",     "11-50",   "Python, TensorFlow, FHIR",
     "Mission-driven; we exist to reduce diagnostic errors.", 2021),
    (6,  "RetailRadar",         "Retail Analytics",        "seed",     "11-50",   "Python, Spark, Snowflake",
     "Obsessed with data quality and fast iteration.", 2022),
    (7,  "EduForge",            "EdTech",                  "seed",     "1-10",    "React, Django, PostgreSQL",
     "Remote-first, async-first. Results over hours.", 2023),

    # Series A (8–14)
    (8,  "Inferix",             "Artificial Intelligence", "series-a", "51-200",  "Python, PyTorch, Kubernetes, MLflow",
     "Fast promotion cycles; engineers own production models end-to-end.", 2020),
    (9,  "Qularis",             "FinTech",                 "series-a", "51-200",  "Java, Spring Boot, Kafka, PostgreSQL",
     "Compliance-first but moves fast inside guardrails.", 2019),
    (10, "LogiLink",            "Supply Chain",            "series-a", "51-200",  "Python, FastAPI, React, MongoDB",
     "Cross-functional squads with full stack ownership.", 2020),
    (11, "Pennant Health",      "HealthTech",              "series-a", "51-200",  "Python, TensorFlow, Django, AWS",
     "Patient outcomes drive every engineering decision.", 2018),
    (12, "CyberNomad Security", "Cybersecurity",           "series-a", "51-200",  "Go, Python, Kubernetes, Rust",
     "Aggressive red-team culture; break things before attackers do.", 2019),
    (13, "GreenGrid Energy",    "CleanTech",               "series-a", "51-200",  "Python, Spark, Airflow, AWS",
     "Flat hierarchy; every engineer attends quarterly OKR reviews.", 2020),
    (14, "UrbanMobility",       "Mobility Tech",           "series-a", "51-200",  "Python, Go, React, PostgreSQL, Redis",
     "High-velocity deployments; 10+ releases per day.", 2021),

    # Series B (15–18)
    (15, "Zeta Payments",       "FinTech",                 "series-b", "201-500", "Java, Kotlin, Kafka, Cassandra, AWS",
     "Engineering excellence with a startup pace.", 2017),
    (16, "Nuvola Cloud",        "Cloud Infrastructure",    "series-b", "201-500", "Go, Kubernetes, Terraform, GCP",
     "Remote-first with quarterly all-hands in Bangalore.", 2016),
    (17, "Arctis Analytics",    "Data Analytics",          "series-b", "201-500", "Python, Spark, dbt, Snowflake, Airflow",
     "Data mesh architecture; every team owns their domain data.", 2017),
    (18, "Fendora Retail",      "Retail Tech",             "series-b", "201-500", "React, Node.js, Python, PostgreSQL, Redis",
     "Product and engineering co-own the roadmap.", 2016),

    # MNC (19–20)
    (19, "TechGiant India",     "Enterprise Software",     "mnc",      "10000+",  "Java, Python, AWS, Kubernetes, Terraform",
     "Structured growth paths; dedicated L&D budget per engineer.", 2000),
    (20, "GlobalSoft Systems",  "IT Services",             "mnc",      "10000+",  "Python, Java, React, Azure, Docker",
     "Work on Fortune 500 client projects with a global team.", 1998),
]

# ── Job data (50 records) ─────────────────────────────────────────────────────

JOBS = [
    # ── ML Engineer × 10 ─────────────────────────────────────────────────────
    (1,  "ML Engineer – Computer Vision",
     "NeuralNest AI",       "Bangalore", 20, 35, 2, 5,
     "Python, PyTorch, OpenCV, Docker, AWS, MLflow",
     "Artificial Intelligence", "seed",     1,
     "Design and ship production CV pipelines for real-time defect detection. "
     "You will own model training, evaluation, and deployment on AWS SageMaker. "
     "Strong Python and PyTorch skills required; OpenCV experience a plus."),

    (2,  "Senior ML Engineer – NLP",
     "Inferix",             "Hyderabad", 35, 55, 5, 9,
     "Python, PyTorch, Hugging Face, Kubernetes, MLflow, SQL",
     "Artificial Intelligence", "series-a", 1,
     "Lead NLP model development for our enterprise search product. "
     "You will fine-tune LLMs, build evaluation harnesses, and push models to K8s. "
     "Solid understanding of transformer architectures required."),

    (3,  "ML Engineer – Recommendations",
     "Fendora Retail",      "Mumbai",    25, 40, 3, 6,
     "Python, TensorFlow, scikit-learn, Spark, SQL, Docker",
     "Retail Tech",         "series-b", 0,
     "Build collaborative filtering and two-tower recommendation models. "
     "Work with petabyte-scale clickstream data on Spark. "
     "Familiarity with A/B testing frameworks expected."),

    (4,  "Junior ML Engineer",
     "BioSignal Tech",      "Pune",       8, 14, 0, 2,
     "Python, scikit-learn, TensorFlow, SQL, Git",
     "HealthTech",          "seed",      0,
     "Entry-level role supporting senior ML engineers in building biosignal "
     "classifiers. Exposure to medical datasets and HIPAA compliance preferred."),

    (5,  "ML Engineer – Time Series",
     "GreenGrid Energy",    "Delhi",     22, 36, 3, 7,
     "Python, PyTorch, scikit-learn, Airflow, AWS, SQL",
     "CleanTech",           "series-a", 1,
     "Develop demand forecasting models for renewable energy grids. "
     "Production experience with time-series forecasting at scale required."),

    (6,  "Staff ML Engineer",
     "Zeta Payments",       "Bangalore", 45, 65, 8, 12,
     "Python, PyTorch, Kafka, Kubernetes, MLflow, Docker, SQL",
     "FinTech",             "series-b", 1,
     "Define ML platform strategy and mentor a team of 6 engineers. "
     "Own fraud detection models processing 50M+ transactions/day. "
     "Strong systems design and stakeholder communication skills required."),

    (7,  "ML Engineer – MLOps",
     "Arctis Analytics",    "Remote",    28, 42, 3, 6,
     "Python, MLflow, Docker, Kubernetes, Airflow, Terraform, AWS",
     "Data Analytics",      "series-b", 1,
     "Build the ML platform that 40 data scientists rely on. "
     "Responsibilities include feature store design, model registry, and CI/CD. "
     "Prior MLOps platform experience strongly preferred."),

    (8,  "ML Engineer – Search",
     "TechGiant India",     "Hyderabad", 30, 50, 4, 8,
     "Python, PyTorch, Elasticsearch, Docker, AWS, SQL",
     "Enterprise Software", "mnc",       0,
     "Improve ranking and retrieval for an enterprise search product serving "
     "millions of users. Experience with dense retrieval (DPR, ColBERT) a plus."),

    (9,  "ML Engineer – Computer Vision",
     "Pennant Health",      "Bangalore", 18, 30, 2, 4,
     "Python, TensorFlow, OpenCV, Docker, AWS",
     "HealthTech",          "series-a", 0,
     "Build medical image segmentation models for radiology workflows. "
     "Experience with DICOM data and clinical AI validation preferred."),

    (10, "ML Engineer – Generative AI",
     "LogiLink",            "Mumbai",    24, 38, 2, 5,
     "Python, PyTorch, Hugging Face, FastAPI, Docker, SQL",
     "Supply Chain",        "series-a", 1,
     "Integrate LLM-powered features into our supply chain platform. "
     "Build RAG pipelines, prompt engineering frameworks, and evaluation suites."),

    # ── Data Scientist × 8 ───────────────────────────────────────────────────
    (11, "Data Scientist – Growth",
     "UrbanMobility",       "Bangalore", 18, 28, 2, 5,
     "Python, SQL, scikit-learn, Pandas, Spark, Tableau",
     "Mobility Tech",       "series-a", 0,
     "Drive growth experimentation using causal inference and propensity modelling. "
     "Collaborate with product to design A/B tests and interpret results."),

    (12, "Senior Data Scientist",
     "Zeta Payments",       "Mumbai",    30, 48, 5, 9,
     "Python, SQL, PySpark, scikit-learn, MLflow, Airflow",
     "FinTech",             "series-b", 0,
     "Build credit risk models that power lending decisions for 2M+ users. "
     "Experience with SHAP, model cards, and regulatory model documentation required."),

    (13, "Data Scientist – NLP",
     "DataSprout",          "Remote",    12, 20, 1, 3,
     "Python, SQL, scikit-learn, spaCy, Hugging Face, Git",
     "Data Analytics",      "seed",      1,
     "Mine customer feedback and support tickets to uncover product insights. "
     "Build text classification and sentiment pipelines. "
     "Great opportunity to grow in a small, fast-paced team."),

    (14, "Data Scientist – Marketing Analytics",
     "RetailRadar",         "Hyderabad", 14, 22, 2, 4,
     "Python, SQL, R, Pandas, scikit-learn, Tableau",
     "Retail Analytics",    "seed",      0,
     "Develop customer segmentation and CLV models for retail clients. "
     "Present insights directly to C-suite stakeholders. "
     "Strong SQL and storytelling skills essential."),

    (15, "Data Scientist – Healthcare",
     "BioSignal Tech",      "Pune",      16, 26, 2, 4,
     "Python, SQL, scikit-learn, TensorFlow, Pandas",
     "HealthTech",          "seed",      0,
     "Analyze patient biosignal data to build early-warning clinical models. "
     "Familiarity with IRB processes and clinical trial data a bonus."),

    (16, "Lead Data Scientist",
     "Arctis Analytics",    "Bangalore", 38, 58, 7, 11,
     "Python, SQL, PySpark, scikit-learn, MLflow, dbt, Airflow",
     "Data Analytics",      "series-b", 1,
     "Lead a team of 5 data scientists and define the analytical roadmap. "
     "Own end-to-end: from hypothesis to production model to business outcome."),

    (17, "Data Scientist – Operations Research",
     "LogiLink",            "Delhi",     20, 32, 3, 6,
     "Python, SQL, PuLP, OR-Tools, Pandas, scikit-learn",
     "Supply Chain",        "series-a", 1,
     "Apply optimization and simulation to solve last-mile logistics problems. "
     "Experience with vehicle routing, warehouse slotting, or scheduling preferred."),

    (18, "Data Scientist – Product",
     "EduForge",            "Remote",    10, 18, 1, 3,
     "Python, SQL, scikit-learn, Pandas, Mixpanel",
     "EdTech",              "seed",      1,
     "Instrument learning outcomes using funnel analysis and cohort studies. "
     "Work directly with the founding team to shape the product roadmap."),

    # ── Backend Engineer × 10 ────────────────────────────────────────────────
    (19, "Backend Engineer – Python",
     "LogiLink",            "Bangalore", 16, 28, 2, 5,
     "Python, FastAPI, PostgreSQL, Redis, Docker, Celery",
     "Supply Chain",        "series-a", 0,
     "Design and build microservices for our real-time shipment tracking platform. "
     "Proficiency in async Python and relational database design required."),

    (20, "Senior Backend Engineer – Go",
     "CyberNomad Security", "Hyderabad", 28, 45, 4, 8,
     "Go, gRPC, PostgreSQL, Redis, Docker, Kubernetes",
     "Cybersecurity",       "series-a", 1,
     "Build high-throughput data ingestion pipelines for threat intelligence. "
     "Go experience mandatory; gRPC and protocol design a strong plus."),

    (21, "Backend Engineer – Java",
     "Qularis",             "Mumbai",    20, 35, 3, 6,
     "Java, Spring Boot, Kafka, PostgreSQL, Docker, Redis",
     "FinTech",             "series-a", 0,
     "Develop the payments processing engine handling 10K TPS. "
     "Experience with distributed transactions and idempotency required."),

    (22, "Junior Backend Engineer",
     "Stackr",              "Remote",     8, 14, 0, 2,
     "Python, Django, PostgreSQL, Git, REST APIs",
     "Developer Tools",     "seed",      1,
     "Build and maintain API endpoints for our developer platform. "
     "Great mentorship environment for early-career engineers."),

    (23, "Backend Engineer – Data Platform",
     "Arctis Analytics",    "Bangalore", 22, 38, 3, 6,
     "Python, FastAPI, PostgreSQL, Spark, Airflow, Docker",
     "Data Analytics",      "series-b", 1,
     "Build APIs that expose our data platform to 200+ internal and external users. "
     "Experience with streaming data and schema evolution a plus."),

    (24, "Senior Backend Engineer – FinTech",
     "Zeta Payments",       "Mumbai",    32, 52, 5, 9,
     "Java, Kotlin, Kafka, Cassandra, gRPC, Docker, Kubernetes",
     "FinTech",             "series-b", 0,
     "Own the ledger service that records every financial transaction. "
     "Strong distributed systems background required; fintech experience preferred."),

    (25, "Backend Engineer – Healthcare APIs",
     "Pennant Health",      "Delhi",     18, 30, 2, 5,
     "Python, FastAPI, PostgreSQL, Docker, AWS, FHIR",
     "HealthTech",          "series-a", 0,
     "Build FHIR-compliant APIs connecting EHR systems and clinical AI tools. "
     "Healthcare interoperability experience strongly preferred."),

    (26, "Backend Engineer – Mobility",
     "UrbanMobility",       "Bangalore", 20, 33, 2, 5,
     "Go, Python, PostgreSQL, Redis, Kafka, Docker",
     "Mobility Tech",       "series-a", 0,
     "Scale the ride-matching backend to handle 500K concurrent sessions. "
     "Experience with geospatial indexing (PostGIS, H3) is a bonus."),

    (27, "Staff Backend Engineer",
     "TechGiant India",     "Hyderabad", 42, 65, 8, 14,
     "Java, Python, AWS, Kubernetes, Kafka, PostgreSQL, gRPC",
     "Enterprise Software", "mnc",       0,
     "Provide technical leadership across 4 backend squads. "
     "Own architecture decisions for a platform serving Fortune 500 clients."),

    (28, "Backend Engineer – Python/AWS",
     "GreenGrid Energy",    "Pune",      18, 30, 2, 5,
     "Python, Django, PostgreSQL, AWS Lambda, SQS, Docker",
     "CleanTech",           "series-a", 1,
     "Build serverless microservices for grid monitoring and alerting. "
     "AWS experience strongly preferred; passion for clean energy a plus."),

    # ── Full Stack Developer × 8 ─────────────────────────────────────────────
    (29, "Full Stack Developer",
     "Stackr",              "Remote",    10, 18, 1, 3,
     "React, Node.js, PostgreSQL, Docker, Git, REST APIs",
     "Developer Tools",     "seed",      1,
     "Build end-to-end features for our CLI and web dashboard. "
     "Equal comfort in React and Node.js required."),

    (30, "Full Stack Developer – EdTech",
     "EduForge",            "Remote",    12, 20, 1, 3,
     "React, Django, Python, PostgreSQL, AWS S3, Git",
     "EdTech",              "seed",      1,
     "Build interactive learning modules and assessment engines. "
     "Experience with video streaming or WebRTC a bonus."),

    (31, "Senior Full Stack Developer",
     "Fendora Retail",      "Mumbai",    24, 40, 4, 7,
     "React, Node.js, Python, PostgreSQL, MongoDB, Docker, Redis",
     "Retail Tech",         "series-b", 0,
     "Lead frontend architecture and own the backend BFF layer. "
     "GraphQL and micro-frontend experience preferred."),

    (32, "Full Stack Developer – FinTech",
     "Qularis",             "Bangalore", 18, 30, 2, 5,
     "React, Java, Spring Boot, PostgreSQL, Docker",
     "FinTech",             "series-a", 0,
     "Build the merchant-facing dashboard and its supporting APIs. "
     "Ability to context-switch between frontend and backend daily required."),

    (33, "Full Stack Developer – HealthTech",
     "Pennant Health",      "Hyderabad", 16, 26, 2, 4,
     "React, Python, FastAPI, PostgreSQL, AWS, Docker",
     "HealthTech",          "series-a", 0,
     "Build patient and clinician-facing dashboards for AI-assisted diagnosis. "
     "Accessibility (WCAG 2.1) knowledge valued."),

    (34, "Junior Full Stack Developer",
     "RetailRadar",         "Pune",       7, 13, 0, 2,
     "React, Node.js, MongoDB, Git, REST APIs",
     "Retail Analytics",    "seed",      0,
     "Assist in building analytics dashboards for retail clients. "
     "Strong fundamentals in JavaScript and REST APIs required."),

    (35, "Full Stack Developer – Mobility",
     "UrbanMobility",       "Delhi",     20, 32, 3, 6,
     "React, Node.js, Python, PostgreSQL, Redis, Docker, Kubernetes",
     "Mobility Tech",       "series-a", 0,
     "Build rider and driver apps backed by high-performance APIs. "
     "Experience with real-time features (WebSockets, SSE) required."),

    (36, "Lead Full Stack Developer",
     "GlobalSoft Systems",  "Bangalore", 30, 50, 6, 10,
     "React, Node.js, Java, PostgreSQL, MongoDB, Docker, Azure",
     "IT Services",         "mnc",       0,
     "Lead a cross-functional team of 8 engineers on a global retail client project. "
     "Strong communication and estimation skills required."),

    # ── DevOps Engineer × 7 ──────────────────────────────────────────────────
    (37, "DevOps Engineer",
     "CloudSeed Labs",      "Bangalore",  9, 16, 1, 3,
     "Docker, Kubernetes, Terraform, AWS, CI/CD, Linux, Python",
     "Cloud Infrastructure","seed",      1,
     "Help build our internal platform for cloud-native deployments. "
     "Exposure to Helm, ArgoCD, and GitOps workflows preferred."),

    (38, "Senior DevOps Engineer",
     "Nuvola Cloud",        "Remote",    26, 42, 4, 8,
     "Docker, Kubernetes, Terraform, GCP, CI/CD, Go, Linux",
     "Cloud Infrastructure","series-b", 1,
     "Own the reliability and scalability of a multi-tenant SaaS platform on GCP. "
     "Experience with SLO-based alerting and chaos engineering valued."),

    (39, "DevOps Engineer – Security",
     "CyberNomad Security", "Hyderabad", 22, 36, 3, 6,
     "Docker, Kubernetes, Terraform, AWS, Linux, Python, Vault",
     "Cybersecurity",       "series-a", 0,
     "Embed security into CI/CD pipelines and harden Kubernetes clusters. "
     "Experience with SAST, DAST, or supply-chain security tooling required."),

    (40, "DevOps Engineer – FinTech",
     "Zeta Payments",       "Mumbai",    24, 40, 3, 7,
     "Docker, Kubernetes, Terraform, AWS, Kafka, Linux, CI/CD",
     "FinTech",             "series-b", 0,
     "Maintain 99.99% uptime for payment infrastructure across 3 AWS regions. "
     "On-call rotation required; PagerDuty experience a plus."),

    (41, "Platform Engineer",
     "Arctis Analytics",    "Bangalore", 28, 45, 4, 8,
     "Kubernetes, Terraform, AWS, Python, Airflow, Docker, CI/CD",
     "Data Analytics",      "series-b", 1,
     "Build and operate the data engineering platform used by 40+ analysts. "
     "Spark cluster management and Airflow administration experience preferred."),

    (42, "Junior DevOps / SRE",
     "LogiLink",            "Pune",       8, 15, 0, 2,
     "Docker, Linux, AWS, CI/CD, Git, Python",
     "Supply Chain",        "series-a", 0,
     "Support senior SREs in building CI/CD pipelines and incident response. "
     "Strong Linux fundamentals and scripting skills required."),

    (43, "Staff DevOps Engineer",
     "TechGiant India",     "Delhi",     40, 62, 8, 13,
     "Kubernetes, Terraform, AWS, GCP, Docker, Go, Linux, CI/CD",
     "Enterprise Software", "mnc",       0,
     "Architect multi-cloud infrastructure for enterprise clients. "
     "FinOps and cost optimisation experience strongly valued."),

    # ── Product Manager × 7 ──────────────────────────────────────────────────
    (44, "Product Manager – AI Products",
     "Inferix",             "Bangalore", 22, 36, 3, 6,
     "Product Strategy, SQL, Data Analysis, Agile, User Research",
     "Artificial Intelligence", "series-a", 1,
     "Define the roadmap for our enterprise AI search product. "
     "Ability to write detailed PRDs and work closely with ML engineers required. "
     "Prior experience in AI/ML product management preferred."),

    (45, "Senior Product Manager",
     "Zeta Payments",       "Mumbai",    30, 50, 5, 9,
     "Product Strategy, SQL, Fintech Regulations, Agile, OKRs",
     "FinTech",             "series-b", 0,
     "Own the checkout and payments product for 2M+ merchants. "
     "Understanding of PCI-DSS and payment flows mandatory."),

    (46, "Product Manager – Growth",
     "UrbanMobility",       "Bangalore", 18, 30, 2, 5,
     "Growth Hacking, SQL, A/B Testing, User Research, Agile",
     "Mobility Tech",       "series-a", 0,
     "Drive rider acquisition and retention through data-driven experimentation. "
     "Comfortable running 10+ simultaneous A/B tests."),

    (47, "Associate Product Manager",
     "EduForge",            "Remote",     8, 14, 0, 2,
     "Product Thinking, SQL, User Research, Figma, Agile",
     "EdTech",              "seed",      1,
     "Great APM programme for early-career PMs. Work directly with the CEO. "
     "Strong analytical thinking and empathy for learners required."),

    (48, "Product Manager – Platform",
     "Nuvola Cloud",        "Hyderabad", 25, 42, 4, 8,
     "Platform Products, APIs, SQL, Developer Experience, Agile",
     "Cloud Infrastructure","series-b", 1,
     "Own the developer platform product used by 3000+ engineering teams. "
     "Prior experience as a software engineer strongly preferred."),

    (49, "Product Manager – Data",
     "Arctis Analytics",    "Bangalore", 22, 38, 3, 6,
     "Data Products, SQL, BI Tools, Stakeholder Management, Agile",
     "Data Analytics",      "series-b", 1,
     "Define and ship data product features across our analytics platform. "
     "Comfortable translating business requirements into data models."),

    (50, "Group Product Manager",
     "GlobalSoft Systems",  "Delhi",     38, 60, 8, 14,
     "Product Strategy, Stakeholder Management, OKRs, Agile, SQL",
     "IT Services",         "mnc",       0,
     "Lead 3 PMs and a portfolio of 5 enterprise software products. "
     "Strong executive communication and roadmap prioritisation skills required."),
]


def build_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(CREATE_COMPANIES)
    cur.execute(CREATE_JOBS)

    cur.executemany(
        "INSERT OR REPLACE INTO companies VALUES (?,?,?,?,?,?,?,?)", COMPANIES
    )
    cur.executemany(
        "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", JOBS
    )

    conn.commit()
    return conn


def print_summary(conn):
    cur = conn.cursor()

    print("\n" + "=" * 60)
    print("  DATABASE SUMMARY")
    print("=" * 60)

    for table in ("companies", "jobs"):
        (count,) = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table:12s}  →  {count} rows")

    print("\n── Role distribution ──────────────────────────────────────")
    rows = cur.execute(
        "SELECT title, COUNT(*) FROM jobs GROUP BY "
        "CASE "
        "  WHEN title LIKE '%ML Engineer%' THEN 'ML Engineer' "
        "  WHEN title LIKE '%Data Scientist%' THEN 'Data Scientist' "
        "  WHEN title LIKE '%Backend Engineer%' OR title LIKE '%Staff Backend%' THEN 'Backend Engineer' "
        "  WHEN title LIKE '%Full Stack%' THEN 'Full Stack Developer' "
        "  WHEN title LIKE '%DevOps%' OR title LIKE '%Platform Engineer%' OR title LIKE '%SRE%' THEN 'DevOps Engineer' "
        "  WHEN title LIKE '%Product Manager%' OR title LIKE '%Group Product%' OR title LIKE '%Associate Product%' THEN 'Product Manager' "
        "  ELSE title END "
        "ORDER BY 2 DESC"
    ).fetchall()
    # Aggregate manually for cleaner output
    from collections import defaultdict
    buckets = defaultdict(int)
    for title, cnt in rows:
        if "ML Engineer" in title or "Staff ML" in title:
            buckets["ML Engineer"] += cnt
        elif "Data Scientist" in title or "Lead Data" in title:
            buckets["Data Scientist"] += cnt
        elif "Backend" in title or "Staff Backend" in title:
            buckets["Backend Engineer"] += cnt
        elif "Full Stack" in title:
            buckets["Full Stack Developer"] += cnt
        elif "DevOps" in title or "Platform Engineer" in title or "SRE" in title:
            buckets["DevOps / SRE"] += cnt
        elif "Product Manager" in title or "Product" in title:
            buckets["Product Manager"] += cnt
        else:
            buckets[title] += cnt
    for role, cnt in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {role:26s}  {cnt}")

    print("\n── Company stage distribution ─────────────────────────────")
    for stage, cnt in cur.execute(
        "SELECT company_stage, COUNT(*) FROM jobs GROUP BY company_stage ORDER BY 2 DESC"
    ).fetchall():
        print(f"  {stage:12s}  {cnt}")

    print("\n── Location distribution ──────────────────────────────────")
    for loc, cnt in cur.execute(
        "SELECT location, COUNT(*) FROM jobs GROUP BY location ORDER BY 2 DESC"
    ).fetchall():
        print(f"  {loc:12s}  {cnt}")

    print("\n── Sample query: remote ML/Data roles paying ≥ 20 LPA ────")
    sample = cur.execute(
        "SELECT title, company, salary_min, salary_max "
        "FROM jobs "
        "WHERE remote_friendly = 1 "
        "  AND (title LIKE '%ML%' OR title LIKE '%Data%') "
        "  AND salary_min >= 20 "
        "ORDER BY salary_max DESC LIMIT 5"
    ).fetchall()
    if sample:
        for row in sample:
            print(f"  {row[0][:38]:38s}  {row[1]:20s}  {row[2]}-{row[3]} LPA")
    else:
        print("  (no rows matched)")

    print("\n── Sample query: seed-stage companies ─────────────────────")
    for row in cur.execute(
        "SELECT name, industry, founded_year FROM companies WHERE stage = 'seed'"
    ).fetchall():
        print(f"  {row[0]:22s}  {row[1]:28s}  founded {row[2]}")

    print("\n" + "=" * 60)
    print(f"  DB path: {DB_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print(f"Creating database at {DB_PATH} …")
    conn = build_db()
    print_summary(conn)
    conn.close()
    print("Done.")
