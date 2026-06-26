"""
Corpus content for the Hiring Co-pilot RAG tool.

Three dicts mapping {filename: text}. build_corpus() in rag_tool.py writes these
to data/corpus/ as .txt files. Kept separate from rag_tool.py so the RAG logic
stays readable — this module is purely data.
"""

# ── 10 company-culture documents ─────────────────────────────────────────────

CULTURE_DOCS = {
    "culture_seed_fintech.txt": """\
COMPANY TYPE: Early-stage fintech startup (Seed)

MISSION & VISION
We are building the financial rails for India's next 200 million users — people
who have a smartphone but have never had real access to credit. The vision is a
single app that underwrites, lends, and collects safely at scale. At seed stage
we are pre-product-market-fit: the product will change three times before it
sticks, and everyone must be comfortable with that.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
Ownership over polish. We hire people who ship a rough thing that works this week
over a perfect thing next quarter. We look for candidates who have built
something end-to-end alone — a side project, a college product, a feature they
drove start to finish. Comfort with ambiguity is non-negotiable: there is no
spec, no PM, and often no second opinion. We value people who ask "should we even
build this?" not just "how do I build this?"

INTERVIEW PROCESS
Three rounds. (1) Founder call — why fintech, why early stage, what have you
built. (2) Practical build round: a 2-3 hour take-home or live session building a
small API or data pipeline; we care about working code and clear tradeoffs, not
algorithm puzzles. (3) Culture + ownership round with the founding team: we probe
a real decision you made under uncertainty and what you'd do differently.

TECH CULTURE
Extremely fast-paced. Deploys multiple times a day, minimal process, direct
access to founders. We optimize for learning speed over correctness of process.
Code review is light; tests cover the money paths only. RBI/compliance reality
means the lending logic must be careful even when everything else is scrappy.

GROWTH OPPORTUNITIES
Title inflation is real and fast — an early engineer can be a team lead within a
year if the company grows. You get enormous surface area: you will touch
infra, product, and data. The flipside is no mentorship structure.

RED FLAGS WE SCREEN FOR
Candidates who need clear instructions to move. People who ask about process,
hierarchy, and work-life balance before asking about the problem. Resume-driven
developers who chase trendy tech over shipping. Anyone who has only worked inside
large, well-defined systems and has never owned ambiguity.
""",

    "culture_seriesa_saas.txt": """\
COMPANY TYPE: Growth-stage B2B SaaS company (Series A)

MISSION & VISION
We sell workflow software to mid-market companies and have found product-market
fit — now the job is to scale revenue 5x without the product collapsing under its
own weight. Series A means we have real customers, real SLAs, and real revenue to
protect, but we are still small enough that one engineer's decision can move the
whole product.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
We look for people who can balance speed with not breaking paying customers. The
seed-stage "move fast and break things" energy must now coexist with on-call,
backwards compatibility, and customer trust. We value engineers who think about
the customer impact of a change, who write enough tests, and who can scope a
feature into shippable increments. Strong written communication matters because
we are partially async.

INTERVIEW PROCESS
Four rounds. (1) Recruiter/role fit. (2) Technical screen — a practical coding
problem close to real work (API design, data modeling). (3) System design scaled
to your level: how would you design multi-tenant feature X. (4) Cross-functional
round with product and a founder on collaboration and customer empathy.

TECH CULTURE
Fast but maturing. We have CI/CD, code review, and staging now. Squad-based with
real ownership of services. Bias toward action but with guardrails. Promotion
cycles are quick because the company is growing under you.

GROWTH OPPORTUNITIES
This is the sweet spot for career acceleration: enough structure to learn good
habits, enough chaos to own big things. Early Series A engineers often become the
first EMs and staff engineers.

RED FLAGS WE SCREEN FOR
Engineers who only optimize for speed and dismiss tests/on-call as "process."
Equally, people so process-heavy they can't ship without a perfect spec.
Candidates who can't explain the business reason behind a technical choice.
""",

    "culture_seriesb_edtech.txt": """\
COMPANY TYPE: Scale-up edtech company (Series B)

MISSION & VISION
We help millions of students learn better through adaptive content. At Series B we
are past survival and into scale: tens of millions of users, large content and
data pipelines, and the operational reality of a consumer product with seasonal
spikes (exam season). Reliability and unit economics now matter as much as growth.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
Engineers who think at scale and care about cost. We look for people who have
operated systems under real load and understand caching, queues, and graceful
degradation. We value mentorship ability — teams are growing and seniors must
lift juniors. Data sensitivity matters: we handle minors' data.

INTERVIEW PROCESS
Four to five rounds. (1) Screen. (2) DS&A round (yes, we still test algorithms at
this scale because hot paths matter). (3) System design at scale — design the
quiz-serving system for 5M concurrent users during exam season. (4) Domain/role
deep-dive. (5) Bar-raiser / values round.

TECH CULTURE
Methodical where it counts (payments, content correctness, student data) and fast
where it's safe (experiments, growth features). Strong observability culture,
incident reviews, and SLOs. More structure than a Series A.

GROWTH OPPORTUNITIES
Defined ladders are emerging. Good place to grow into a senior/staff IC or an EM
with real scope. Internal mobility across content, platform, and growth teams.

RED FLAGS WE SCREEN FOR
People who over-engineer for scale they don't have yet, and people who ignore cost
entirely. Candidates who can't handle a system design question that has no clean
answer. Lack of empathy for the student/end-user.
""",

    "culture_mnc_it_services.txt": """\
COMPANY TYPE: Large Indian MNC IT services company

MISSION & VISION
We deliver technology services and digital transformation to Fortune 500 clients
across the globe. Our strength is scale, process maturity, and the ability to
staff and deliver large, multi-year programs reliably. Stability and client trust
are the core of the brand.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
Reliability, communication, and the ability to work within structured processes
and large teams. Client-facing roles require professionalism and clear
communication with non-technical stakeholders. We value certifications, process
discipline (Agile/SAFe), and the ability to ramp onto a client's tech stack
quickly. Individual heroics matter less than consistent, documented delivery.

INTERVIEW PROCESS
Structured and standardized. (1) Aptitude/online assessment (for many roles).
(2) Technical round on fundamentals and the relevant stack. (3) Managerial round
on project experience, process, and behavioral questions. (4) HR round on
logistics, notice period, and compensation. Process is consistent across
candidates by design.

TECH CULTURE
Methodical and process-driven. Defined SDLC, documentation, change management, and
approvals. Slower-moving than startups but predictable. Technology choices are
often dictated by the client. Strong emphasis on training and certifications.

GROWTH OPPORTUNITIES
Clear, predictable career ladder with defined bands and dedicated L&D budget.
Excellent for structured learning, certifications, and exposure to enterprise-scale
systems and global clients. Lateral movement across domains is possible.

RED FLAGS WE SCREEN FOR
Job hoppers who won't stay through a project cycle. Candidates who disparage
process or documentation. Poor communication or inability to work in large teams.
Over-focus on "cool tech" over client needs.
""",

    "culture_us_product_india.txt": """\
COMPANY TYPE: US-based product company, India engineering office (GCC/captive)

MISSION & VISION
We are the India engineering center of a US product company. The India office is
not a vendor — it owns full products and services end-to-end, with the same bar as
the US teams. The mission is to build world-class products while operating as one
global engineering org.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
A high engineering bar comparable to top US companies: strong fundamentals, clean
code, rigorous system design, and excellent written communication for async work
across time zones. We look for ownership and the maturity to make decisions
without waiting for the US team to wake up. Product sense and the ability to push
back on requirements are valued.

INTERVIEW PROCESS
Rigorous, US-style. (1) Recruiter screen. (2) One or two DS&A rounds (real
algorithmic depth). (3) System design round at senior levels. (4) Behavioral /
"hiring manager" round on ownership and collaboration. (5) Sometimes a
bar-raiser. The bar is high and consistent globally.

TECH CULTURE
Strong engineering excellence: code review, testing, design docs, on-call done
well. More methodical than a startup but faster and flatter than an MNC services
firm. Heavy async communication and documentation because of the time-zone gap.

GROWTH OPPORTUNITIES
Top-tier compensation for India, strong mentorship, and the ability to grow on a
global ladder. Exposure to large-scale production systems and world-class peers.
Visibility to the US org matters for promotion.

RED FLAGS WE SCREEN FOR
Weak fundamentals masked by framework familiarity. Poor written communication.
Candidates who need constant direction. Inability to handle the async, self-driven
working style.
""",

    "culture_gaming_startup.txt": """\
COMPANY TYPE: Gaming startup

MISSION & VISION
We build mobile games that millions play daily. The mission is to create games
that are genuinely fun and retain players for years. We live at the intersection
of creativity, real-time systems, and live-ops economics.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
Passion for games is a real filter — people who play and analyze games build
better ones. We look for engineers comfortable with real-time, low-latency
systems, performance optimization, and the unglamorous reality of live-ops (events,
A/B tests, economy tuning). Creativity and a data-driven mindset together.

INTERVIEW PROCESS
Three to four rounds. (1) Screen + "what games do you play and why are they good."
(2) Technical round emphasizing performance, concurrency, and sometimes a small
game-logic problem. (3) System design for real-time multiplayer or live-ops infra.
(4) Culture round on creativity and handling live-game crises.

TECH CULTURE
Fast and intense, with crunch around launches and live events. Highly metrics-
driven (retention, ARPU, DAU). Tolerance for experimentation and rapid iteration.
Cross-functional with designers and artists.

GROWTH OPPORTUNITIES
Broad: you touch client, server, infra, and data. Fast growth if you can ship hits.
Strong sense of impact when millions play what you built.

RED FLAGS WE SCREEN FOR
Engineers with no interest in games (retention suffers). People who can't handle
the intensity of live-ops or launch crunch. Over-engineering for scale before a
game proves retention. Ignoring performance on low-end devices.
""",

    "culture_healthtech_startup.txt": """\
COMPANY TYPE: HealthTech startup

MISSION & VISION
We are improving patient outcomes through technology — diagnostics, care delivery,
or clinical workflows. The mission is genuinely life-affecting, which raises the
stakes: bugs can harm patients, and trust with clinicians and regulators is
everything.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
Conscientiousness and care. We look for engineers who understand that "move fast
and break things" does not apply when patient safety is involved. Mission
alignment is a strong signal — people drawn to healthcare tend to stay and care.
Comfort with regulatory constraints (HIPAA-equivalent, clinical validation) and
working closely with doctors and domain experts.

INTERVIEW PROCESS
Three to four rounds. (1) Screen + motivation for healthcare. (2) Technical round
relevant to the role. (3) A round with a clinical/domain stakeholder on how you
handle correctness, edge cases, and ambiguity in medical data. (4) Values round
on responsibility and ethics.

TECH CULTURE
A deliberate blend: fast on product surfaces, methodical and careful on anything
touching clinical decisions, patient data, or compliance. Strong emphasis on data
privacy, auditability, and validation.

GROWTH OPPORTUNITIES
Deep domain expertise that compounds — health domain knowledge is rare and
valuable. Mission-driven teams with low churn. Opportunity to shape regulated
products.

RED FLAGS WE SCREEN FOR
Cavalier attitudes toward correctness, testing, or data privacy. Pure mercenaries
with no mission interest (high churn risk). Engineers who can't collaborate with
non-technical clinical experts.
""",

    "culture_b2b_enterprise.txt": """\
COMPANY TYPE: B2B enterprise software company

MISSION & VISION
We build software that large enterprises run their core operations on. The mission
is reliability, security, and deep functionality for complex organizations. Sales
cycles are long, customers are demanding, and contracts are large — so trust and
robustness beat flashy features.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
Engineers who care about correctness, backwards compatibility, security, and the
long tail of enterprise edge cases. We value people who can navigate complexity and
legacy, who document well, and who understand that enterprise customers cannot
tolerate breaking changes. Patience and rigor over novelty.

INTERVIEW PROCESS
Four rounds. (1) Screen. (2) Technical round on solid engineering and debugging of
complex systems. (3) System design emphasizing reliability, security, integrations,
and backwards compatibility. (4) Behavioral round on handling demanding customers
and long-running projects.

TECH CULTURE
Methodical and quality-focused. Strong testing, security review, and release
discipline. Slower iteration than consumer products because stability is the
product. Lots of integrations and configurability.

GROWTH OPPORTUNITIES
Deep technical mastery of complex domains and distributed systems. Stable, senior
IC and architect tracks. Valuable, transferable expertise in security and
reliability.

RED FLAGS WE SCREEN FOR
"Move fast and break things" engineers who dismiss backwards compatibility and
security. People who get bored by complexity and edge cases. Poor documentation
habits.
""",

    "culture_consumer_internet.txt": """\
COMPANY TYPE: Consumer internet company

MISSION & VISION
We build a consumer app used by tens of millions daily — commerce, social, or
content. The mission is delightful, fast experiences at massive scale. Growth,
engagement, and retention metrics drive almost everything.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
Product sense and a data-driven mindset. We look for engineers who care about user
experience, latency, and metrics, and who can run and interpret A/B tests. Comfort
with very large scale, high traffic, and the messiness of real user behavior.
Speed of iteration matters.

INTERVIEW PROCESS
Four rounds. (1) Screen. (2) DS&A round (scale makes algorithms matter). (3)
System design for high-scale, low-latency consumer systems (feed, search, cart).
(4) Product/behavioral round — how you'd improve a metric, how you reason about
user impact.

TECH CULTURE
Fast-paced and experiment-heavy. Ship, measure, iterate. Strong A/B testing and
analytics culture. High-scale infrastructure challenges. Bias toward action with
data as the referee.

GROWTH OPPORTUNITIES
Massive scale exposure and strong product-engineering crossover. Fast growth for
those who move metrics. Good comp and brand value.

RED FLAGS WE SCREEN FOR
Engineers indifferent to product and users ("just give me the spec"). People who
can't reason about scale or latency. Analysis paralysis — over-thinking instead of
experimenting. Ignoring metrics.
""",

    "culture_deeptech_ai.txt": """\
COMPANY TYPE: Deep tech / AI research company

MISSION & VISION
We push the frontier — novel models, hard research problems, and technology that
doesn't exist yet. The mission blends research and product: papers and products
ship together. Timelines are uncertain because real research is uncertain.

WHAT WE ACTUALLY LOOK FOR (BEYOND SKILLS)
Depth and first-principles thinking. We look for people who understand the math and
internals, not just API calls — those who can read papers, reproduce them, and
extend them. Research taste (knowing which problems matter) and the rigor to run
clean experiments. Strong engineering to turn research into something real.

INTERVIEW PROCESS
Three to five rounds. (1) Screen on background and research interests. (2) Deep
technical round on ML fundamentals, math, and your past work — expect to defend
choices. (3) A research/coding exercise: reproduce or extend a result, or design an
experiment. (4) Discussion of a paper or your research. (5) Culture/research-fit.

TECH CULTURE
Methodical on research rigor, but tolerant of exploration and failure (most
experiments fail). Less process around shipping; more around experimental
correctness and reproducibility. Intellectually intense; peers are often PhDs.

GROWTH OPPORTUNITIES
Become a genuine expert at the frontier. Publication opportunities and work with
top researchers. High scarcity value if the bets pay off.

RED FLAGS WE SCREEN FOR
"Wrapper" engineers who only call APIs without understanding internals. People who
need certainty and clear timelines. Inability to read or reproduce papers.
Overclaiming results without rigorous evaluation.
""",
}


# ── 15 skill-context documents ───────────────────────────────────────────────

SKILL_DOCS = {
    "skill_production_ml_experience.txt": """\
TOPIC: What "production ML experience" actually means

When a JD says "production ML experience," it rarely means training models in a
notebook. It means you have owned a model that real users or systems depend on,
through its full lifecycle:

- Data pipelines: getting reliable training and serving data, handling drift,
  missing values, and schema changes — not a clean Kaggle CSV.
- Serving: deploying a model behind an API or batch job with latency, throughput,
  and uptime constraints. Knowing the difference between a 50ms and 500ms model.
- Monitoring: detecting data drift, model decay, and silent failures in
  production; having dashboards and alerts.
- Retraining and versioning: model registries, reproducible training, rollback.
- Evaluation in the real world: offline metrics vs online metrics, A/B testing,
  and the gap between them.

What it does NOT mean: getting 99% accuracy on a benchmark, finishing a course, or
fine-tuning once in Colab. Interviewers probe this by asking "what broke in
production and how did you find out?" Candidates who only have notebook experience
struggle with that question. At startups, "production ML" often also means you
built the infra yourself; at big companies it may mean you used an internal
platform. Be specific about which.
""",

    "skill_strong_fundamentals.txt": """\
TOPIC: What "strong fundamentals" means at different company types

"Strong fundamentals" is one of the most loaded phrases in tech JDs, and it means
different things by company type:

- MNC / IT services: solid grasp of OOP, data structures, the language's standard
  library, SQL, and SDLC process. Tested via standard DS&A and definitions.
- Product companies / US-India offices: deep DS&A (you can solve medium/hard
  algorithm problems), strong systems understanding (memory, concurrency,
  networking), and clean code. The bar is high and tested rigorously.
- Startups (seed/Series A): fundamentals mean you can build things correctly
  without hand-holding — you understand HTTP, databases, indexing, and can debug a
  production issue from first principles. Less algorithm trivia, more "can you make
  it work and not fall over."
- Deep tech / AI: fundamentals extend to math (linear algebra, probability,
  optimization) and the internals of the models/systems you use.

The common thread: fundamentals = understanding WHY things work, not memorizing
HOW to use a framework. A red flag for all of them is someone who can use a tool
but cannot explain what it does underneath.
""",

    "skill_mlops_vs_ml_engineering.txt": """\
TOPIC: Difference between MLOps and ML Engineering

These overlap but are not the same, and JDs often conflate them.

ML Engineering: building the models and the code around them. Feature engineering,
model architecture, training pipelines, evaluation, and shipping a model that
performs well on the task. An ML Engineer is judged on whether the model is good
and works in production.

MLOps: building the platform and automation that lets many models be trained,
deployed, monitored, and retrained reliably. CI/CD for models, feature stores,
model registries, experiment tracking, serving infrastructure, drift monitoring,
and reproducibility. An MLOps engineer is judged on whether the ML system is
reliable, repeatable, and scalable — often they don't build the models themselves.

In practice: a small startup wants one person who does both. A larger company
separates them: ML Engineers/Scientists build models, MLOps/Platform engineers
build the rails. If a JD lists heavy infra (Kubernetes, Terraform, Airflow,
feature stores) under an "ML Engineer" title, it's really an MLOps/platform role.
If it emphasizes modeling, metrics, and experimentation, it's true ML engineering.
Read the responsibilities, not just the title.
""",

    "skill_devops_indian_startups.txt": """\
TOPIC: What DevOps skills mean in Indian startups

"DevOps" in an Indian startup JD is broad and often means "the person who keeps
everything running and automated," not a narrow specialty. Expect:

- Cloud: AWS (most common), sometimes GCP/Azure. EC2, S3, RDS, IAM, VPC basics.
- Containers & orchestration: Docker always; Kubernetes increasingly, though many
  early startups run on simpler setups (ECS, managed services) and only adopt K8s
  when they actually need it.
- CI/CD: GitHub Actions / GitLab CI / Jenkins — building pipelines from scratch.
- IaC: Terraform is the common ask; some use Ansible.
- Observability: setting up logging, metrics, and alerts (Prometheus/Grafana,
  CloudWatch, Datadog).
- Linux and scripting (Bash/Python) are assumed.

The reality: at seed/Series A, a DevOps engineer often also does on-call, cost
optimization, security basics, and a bit of backend. Pure "I only write Terraform"
profiles don't fit. At Series B+ the role narrows and deepens (platform/SRE). A
common gap: candidates know the tools in isolation but have never built a pipeline
or debugged a production outage end-to-end, which is what startups actually need.
""",

    "skill_pytorch_vs_tensorflow.txt": """\
TOPIC: PyTorch vs TensorFlow — when it matters and when it doesn't

For most ML roles, PyTorch vs TensorFlow is NOT a real barrier, and good
interviewers know this. The concepts transfer directly: tensors, autograd,
layers, loss functions, optimizers, training loops, and deployment all map
between the two. An engineer strong in one can become productive in the other in
days to a couple of weeks.

When it genuinely matters:
- Existing codebase: a team with a large TensorFlow/TF-Serving stack reasonably
  prefers someone who won't slow down for weeks. Same for a PyTorch shop.
- Specific ecosystem tools: TF Lite / TF.js for edge/browser, or PyTorch's
  research ecosystem (Hugging Face, Lightning) for fast experimentation.
- Research vs production legacy: research has largely consolidated on PyTorch;
  some older production systems remain on TensorFlow.

When it doesn't matter: most roles, especially if you understand the fundamentals.
A JD listing "TensorFlow" often just reflects what the team uses today. If you
know PyTorch deeply, say so and emphasize that the underlying skills transfer —
strong candidates are rarely rejected over the framework alone. The deeper signal
interviewers want is whether you understand what the framework is doing, not which
one you've typed more.
""",

    "skill_full_stack_startup_vs_mnc.txt": """\
TOPIC: What "full stack" means at a startup vs an MNC

"Full stack developer" describes very different jobs depending on the company.

At a startup: it literally means you build everything — frontend (React), backend
(APIs, business logic), database design, basic DevOps (deploy, monitor), and
sometimes mobile. You own features end-to-end with little specialization. The
expectation is breadth, speed, and comfort context-switching daily. You are
trusted to make decisions across the stack.

At an MNC / large company: "full stack" is narrower and more structured. You work
within an established architecture, frontend and backend are often separate
concerns even if you touch both, and many cross-cutting concerns (infra, security,
DBA, release) are owned by dedicated teams. "Full stack" here often means "comfortable
in both React and a backend framework within our defined patterns," not "owns the
entire system."

Implications for candidates: a startup full-stack role rewards generalists who
ship; an MNC role rewards people who work well within structure and process. Depth
expectations differ too — startups tolerate "good enough across the board," while
larger companies may expect more polish in your primary area. Read whether the JD
emphasizes ownership/breadth (startup) or fitting into a system (MNC).
""",

    "skill_system_design_by_level.txt": """\
TOPIC: System design expectations by experience level

System design rounds scale their expectations with seniority:

- Fresher / 0-1 yr: usually no full system design. You may get basic questions —
  design a URL shortener at a high level, or explain how a web request flows. The
  bar is clear thinking and fundamentals (DB vs cache, API basics).
- Junior / 1-3 yrs: expected to design a single service or feature: data model,
  API, basic scaling (caching, indexing), and obvious failure modes. Not expected
  to design a whole distributed platform.
- Mid / 3-6 yrs: design a multi-component system with real tradeoffs — sharding,
  queues, consistency vs availability, caching strategy, and back-of-envelope
  capacity estimation. Expected to drive the conversation and justify choices.
- Senior / 6-10 yrs: lead an ambiguous, open-ended design end-to-end, including
  non-functional requirements (reliability, cost, security, observability),
  migration paths, and team/operational implications.
- Staff+ : design at the level of multiple systems and org impact; tradeoffs
  across teams, long-term evolution, and "should we build this at all."

A common mismatch: junior candidates over-engineer (premature microservices),
while some senior candidates under-justify. Interviewers care about reasoning and
tradeoffs far more than naming technologies.
""",

    "skill_communication_skills.txt": """\
TOPIC: What "communication skills" means in a technical JD

"Strong communication skills" in a technical JD is not about being extroverted or
fluent in English small talk. It specifically means:

- Written clarity: can you write a clear design doc, PR description, incident
  report, or Slack update that others can act on without a meeting? This is the #1
  meaning, especially in remote/async and US-India teams.
- Explaining technical decisions: can you justify a tradeoff to both engineers and
  non-technical stakeholders (PMs, clients) at the right altitude?
- Asking good questions: surfacing ambiguity early instead of building the wrong
  thing silently.
- Disagreeing constructively: pushing back on a bad requirement with reasons, not
  ego.
- Knowing your audience: technical depth with peers, business framing with
  leadership and clients.

By company type: MNCs and client-facing roles emphasize stakeholder and verbal
communication. Product companies and remote teams emphasize crisp written
communication and design docs. The hidden test in interviews is often the
behavioral round and how clearly you explain your past projects. Rambling, jargon
without structure, or inability to explain WHY are the red flags.
""",

    "skill_leadership_in_ic_roles.txt": """\
TOPIC: Leadership expectations in individual-contributor (IC) roles

Senior IC JDs often list "leadership" without a management title, which confuses
candidates. For an IC, leadership means influence without authority:

- Technical leadership: setting direction on architecture, driving design reviews,
  and raising the engineering bar through example and code review.
- Mentorship: growing junior engineers, unblocking them, and reviewing their work
  constructively — even though they don't report to you.
- Ownership of ambiguous problems: taking a vague, cross-team problem and driving
  it to a solution, coordinating people without managing them.
- Project leadership: breaking down work, sequencing it, and keeping a small team
  aligned and shipping.
- Influence: getting buy-in for technical decisions through clear reasoning and
  trust, not org power.

What it does NOT mean for an IC: people management, performance reviews, or hiring
authority (that's the EM track). A staff/principal IC is expected to have org-wide
technical influence. Interviewers probe this with "tell me about a time you drove
something across teams" or "how did you grow a junior engineer." Candidates who
only describe their own coding output, with no influence on others, read as junior
regardless of years of experience.
""",

    "skill_remote_work_culture.txt": """\
TOPIC: Remote work culture differences

"Remote-friendly" means very different things, and reading it correctly avoids
mismatched expectations:

- Fully remote / remote-first: the company is built around async work — strong
  written communication, documented decisions, flexible hours, and few mandatory
  in-person events. Common at some startups and global product companies.
- Hybrid: typically 2-3 days in office, often with location requirements near a
  hub city. "Remote-friendly" in many Indian JDs actually means hybrid.
- Remote-tolerated: the company is office-centric but allows occasional remote.
  Decisions still happen in hallways; remote workers can be at a disadvantage for
  visibility and promotion.

What thrives remotely: self-driven people with strong written communication who
don't need constant supervision. What struggles: those who learn best by osmosis,
need real-time collaboration, or are early-career and benefit from in-person
mentorship.

Implications: remote roles weight written communication and autonomy heavily in
hiring. Ask explicitly about meeting/time-zone expectations, especially for
US-India roles where late-evening overlap calls are common. A JD that says
"remote" but lists a city is usually hybrid — clarify before assuming.
""",

    "skill_fast_learner_signal.txt": """\
TOPIC: What "fast learner" really signals in a JD

When a JD emphasizes "fast learner" or "ability to pick up new technologies
quickly," it is usually signaling something specific about the environment:

- The tech stack will change, or you'll work on unfamiliar things. Common at
  startups (stack evolves fast) and services companies (you switch client stacks).
- They are NOT looking for someone who only knows the exact listed technologies;
  they will hire for aptitude over checklist match. This is good news for
  candidates missing one or two listed skills.
- They may not have time for long onboarding — you're expected to ramp yourself
  using docs and code, not formal training.

What it asks of you: comfort with ambiguity, self-directed learning, and
transferring concepts across tools (e.g., one cloud to another, one framework to
another). In interviews, demonstrate this with a concrete story: "I had never used
X; here's how I became productive in two weeks and what I shipped."

Subtext to watch: heavy emphasis on "fast learner" plus a very broad skills list
can also signal under-staffing — one person expected to cover a lot. Not
necessarily bad, but ask about team size and support.
""",

    "skill_reading_between_jd_lines.txt": """\
TOPIC: How to read between the lines of a JD

Job descriptions encode signals beyond the literal text. Common translations:

- "Wear many hats" / "fast-paced" / "wear multiple roles" → small team, broad
  scope, limited process, possible long hours. Startup reality.
- "Rockstar / ninja / 10x" → vague role, possibly chaotic, sometimes a warning.
- Very long skills list → either they don't know exactly what they want, or one
  person covers a lot. Don't be scared off if you match ~60-70%.
- "Must be willing to work in a dynamic environment" → expect change and
  ambiguity.
- "Competitive salary" with no number → often means market or below; they want you
  to name a figure first.
- "Immediate joiners preferred" → urgent backfill or growth pressure.
- Emphasis on "ownership" and "self-starter" → little hand-holding; good for
  autonomy, bad if you want structure.
- Heavy process/certification language (Agile, SAFe, ITIL) → larger, structured
  org.
- "Stakeholder management" → client- or cross-team-facing, communication-heavy.

Rule of thumb: required vs preferred matters — apply if you meet most "required"
even if you miss "preferred." And the responsibilities section is more honest than
the buzzword-laden "about us" section.
""",

    "skill_competitive_salary_meaning.txt": """\
TOPIC: What "competitive salary" actually means

"Competitive salary" is one of the least informative phrases in a JD. In practice
it usually means one of:

- Market rate: they pay roughly what comparable companies pay — neither a premium
  nor below. Most common.
- "We don't want to anchor": they prefer you state your expectation first, hoping
  to pay at or below your number.
- Below-market with non-cash upside: early startups sometimes use "competitive"
  while actually offering less cash but equity (see equity-vs-salary tradeoffs).

What it does NOT reliably mean: top-of-market. Companies that pay top quartile
usually say so or signal it (named bands, "top 10% comp," strong Glassdoor data).

How to handle it: research the real range for the role, level, and city
(Bangalore/Hyderabad/Mumbai differ) using Glassdoor, levels.fyi, and peers. Let
the company give a range first if you can. For Indian roles, clarify whether the
quoted CTC includes variable pay, joining bonus, and ESOPs — "competitive CTC" can
hide a large variable or paper-equity component. Always separate fixed cash from
variable and equity when comparing offers.
""",

    "skill_equity_vs_salary.txt": """\
TOPIC: Equity vs salary tradeoffs at different stages

Compensation mixes cash and equity differently by company stage, and the equity is
worth very different amounts:

- Seed: lower cash, larger equity percentage — but the equity is high-risk and
  most likely worth zero. Take it for the learning and upside lottery, not the
  cash. Understand it's a bet.
- Series A/B: moderate cash (closer to market) plus meaningful equity that is less
  risky than seed but still illiquid for years. Reasonable balance for many.
- Late-stage / pre-IPO: strong cash plus equity (RSUs or options) with a clearer,
  though still uncertain, path to liquidity.
- MNC / large public company: highest reliable cash, equity is liquid RSUs (real
  money), lower upside multiple but low risk.

Key things to evaluate in an equity offer: ESOP vs RSU, strike price, vesting
schedule and cliff (typically 4 years, 1-year cliff), the latest valuation and
your percentage, and most importantly the exercise terms and tax treatment in
India (ESOPs are taxed at exercise and again at sale). Treat startup equity as
optional upside, not guaranteed comp. Never accept a big pay cut purely on a paper
equity promise unless you believe in the company and can afford the bet.
""",

    "skill_culture_fit_vs_skill_fit.txt": """\
TOPIC: Culture fit vs skill fit — how companies weigh them

Companies hire on both skill fit (can you do the work) and culture/values fit (will
you thrive and work well here). How they weigh them varies:

- Startups (seed/Series A): culture and ownership fit can outweigh exact skills.
  They'd rather hire an adaptable, high-ownership generalist who's missing a tool
  than a perfect-skills person who needs structure. Attitude and slope > current
  knowledge.
- Product / US-India offices: high skill bar AND strong "values"/behavioral bar —
  both must clear. A brilliant engineer who fails the collaboration bar is often
  rejected ("no jerks" rule).
- MNC / services: skill and process fit dominate; "culture fit" is more about
  professionalism and ability to work in structured teams.
- Deep tech: skill/research depth dominates, but research-collaboration fit matters.

"Culture fit" has a dark side: it can mask bias ("not like us"). Good companies use
"values fit" or "culture add" with structured behavioral interviews instead of vibes.

For candidates: skill fit gets you in the door; culture/values fit often decides
between two qualified candidates. In behavioral rounds, show alignment with how the
company actually works (e.g., ownership for startups, rigor for product companies)
rather than generic enthusiasm.
""",
}


# ── 10 interview-experience documents ────────────────────────────────────────

INTERVIEW_DOCS = {
    "interview_ml_seriesa.txt": """\
INTERVIEW EXPERIENCE: ML Engineer at a Series A startup

ROUNDS & STRUCTURE
Four rounds over two weeks. (1) Recruiter screen (30 min). (2) Technical phone
screen — coding + ML basics (1 hr). (3) Take-home: build a small model + a serving
endpoint on a provided dataset (4-6 hrs of work). (4) Onsite (virtual): take-home
deep-dive, system design for an ML feature, and a founder culture round.

SPECIFIC QUESTIONS ASKED
- "Walk me through your take-home: why this model, what did you trade off?"
- "Your endpoint takes 400ms — how would you get it under 100ms?"
- "How would you detect that this model is degrading in production?"
- System design: "Design a recommendation feature for our product, end to end,
  including data, training, serving, and monitoring."
- Founder round: "Tell me about something you shipped with no clear spec."

WHAT THE INTERVIEWER EMPHASIZED
End-to-end ownership and pragmatism. They cared far more about whether the model
worked in production and whether I understood tradeoffs than about exotic model
choices. They repeatedly pushed on monitoring and failure modes.

WHAT CANDIDATES WISH THEY HAD PREPARED
Serving/latency and monitoring, not just modeling. Many ML candidates over-prepare
training and under-prepare deployment, drift, and "what breaks in prod."

OUTCOME: Offer made. The deciding factors were a clean, working take-home with
sensible tradeoffs and a strong founder-round story about shipping under ambiguity.
""",

    "interview_ds_mnc.txt": """\
INTERVIEW EXPERIENCE: Data Scientist at a large MNC

ROUNDS & STRUCTURE
Standardized, four rounds. (1) Online assessment: aptitude + basic statistics +
SQL. (2) Technical round: statistics, ML concepts, SQL queries, and a case. (3)
Managerial round: past projects, stakeholder handling, process. (4) HR: notice
period, compensation, logistics.

SPECIFIC QUESTIONS ASKED
- "Explain p-value and confidence interval to a non-technical stakeholder."
- "Write a SQL query for month-over-month retention."
- "Difference between bagging and boosting; when use which?"
- Case: "A client's churn went up 5%. How do you investigate?"
- Managerial: "How did you handle a stakeholder who disagreed with your analysis?"

WHAT THE INTERVIEWER EMPHASIZED
Fundamentals, clear communication to non-technical stakeholders, and process. The
MNC valued the ability to explain and to work within a structured delivery model
over cutting-edge modeling. SQL and statistics rigor were tested hard.

WHAT CANDIDATES WISH THEY HAD PREPARED
Crisp business communication and SQL. Strong modelers sometimes fumbled explaining
results simply or writing clean SQL under time pressure.

OUTCOME: Offer made. Strong fundamentals, clear stakeholder communication, and a
stable profile (no job-hopping concerns) sealed it.
""",

    "interview_backend_seed.txt": """\
INTERVIEW EXPERIENCE: Backend Engineer at a seed-stage startup

ROUNDS & STRUCTURE
Lean, three rounds, fast (offer in a week). (1) Founder call. (2) Live build
session: build a small REST API with a database, live, in 90 minutes. (3) Culture
+ ownership chat with the founding team.

SPECIFIC QUESTIONS ASKED
- "Build an API for X with these two endpoints; we'll add a requirement midway."
  (They changed the spec on purpose to see how I adapted.)
- "How would you handle idempotency for this payment endpoint?"
- "Your DB query is slow at 10k rows — what do you check?"
- Founder: "What have you built completely on your own, start to finish?"

WHAT THE INTERVIEWER EMPHASIZED
Shipping working code fast, handling changing requirements without panicking, and
real ownership. No algorithm puzzles. They cared about practical backend judgment
(indexing, idempotency, error handling) and whether I could operate without a spec.

WHAT CANDIDATES WISH THEY HAD PREPARED
Practical, from-scratch building under time pressure. Candidates used to large
codebases with scaffolding struggled to build a service from zero quickly.

OUTCOME: Offer made. The deciding factor was calm adaptation when they changed the
spec mid-build, plus a side project I had shipped solo.
""",

    "interview_fullstack_product.txt": """\
INTERVIEW EXPERIENCE: Full Stack Developer at a product company

ROUNDS & STRUCTURE
Four rounds. (1) Recruiter screen. (2) DS&A coding round (medium difficulty). (3)
Practical full-stack round: build a small feature with a React frontend and an API
backend. (4) System design + behavioral.

SPECIFIC QUESTIONS ASKED
- DS&A: a medium array/hashmap problem with follow-ups on complexity.
- Practical: "Add this feature end to end — UI, API, and data model — and handle
  the error states."
- System design: "Design the frontend and backend for a commenting system with
  pagination and real-time updates."
- Behavioral: "Tell me about a disagreement with a teammate on a technical choice."

WHAT THE INTERVIEWER EMPHASIZED
Genuine comfort across the stack, clean code, and handling edge/error states (not
just the happy path). They valued someone who thinks about UX and API contracts
together, plus reasonable algorithmic ability.

WHAT CANDIDATES WISH THEY HAD PREPARED
Both depth and breadth: some strong frontend devs were weak on API/data modeling,
and some backend devs struggled with React state. Also, error handling and edge
cases, which interviewers probed heavily.

OUTCOME: Offer made. Balanced full-stack ability and careful handling of edge cases
stood out; the DS&A round was passed but was not the deciding factor.
""",

    "interview_devops_seriesb.txt": """\
INTERVIEW EXPERIENCE: DevOps Engineer at a Series B scale-up

ROUNDS & STRUCTURE
Four rounds. (1) Screen. (2) Technical round on Linux, networking, and cloud
fundamentals. (3) Hands-on/scenario round: debug a broken deployment, design a
CI/CD pipeline. (4) System design for reliability + behavioral on incident
handling.

SPECIFIC QUESTIONS ASKED
- "A pod keeps crashing in Kubernetes — walk me through your debugging."
- "Design a CI/CD pipeline for a microservices app with safe rollbacks."
- "How do you achieve zero-downtime deploys?"
- "Describe an outage you handled: detection, mitigation, root cause, prevention."
- Terraform: "How do you manage state across a team safely?"

WHAT THE INTERVIEWER EMPHASIZED
Real operational experience over tool name-dropping. They probed actual debugging
and incident response, observability (SLOs, alerts), and safe deployment practices.
At Series B, reliability and on-call maturity mattered a lot.

WHAT CANDIDATES WISH THEY HAD PREPARED
Concrete incident stories and hands-on debugging. Candidates who knew tools only
theoretically (could describe Kubernetes but never debugged a real cluster issue)
fell short.

OUTCOME: Offer made. Strong, specific incident-response stories and a solid
pipeline design were decisive.
""",

    "interview_pm_consumer.txt": """\
INTERVIEW EXPERIENCE: Product Manager at a consumer internet company

ROUNDS & STRUCTURE
Five rounds. (1) Recruiter screen. (2) Product sense / design round. (3) Analytical
/ metrics round. (4) Execution / prioritization round. (5) Cross-functional +
leadership/behavioral.

SPECIFIC QUESTIONS ASKED
- Product sense: "How would you improve the app's onboarding for first-time users?"
- Metrics: "DAU is flat but installs are up — diagnose what's happening."
- "Define the success metric for feature X and how you'd run the experiment."
- Prioritization: "You have 5 features and one sprint — how do you decide?"
- Behavioral: "Tell me about a product decision you got wrong."

WHAT THE INTERVIEWER EMPHASIZED
Data-driven product thinking, crisp prioritization with clear reasoning, and user
empathy. They wanted structured frameworks but penalized rigid framework-dumping
without insight. Comfort with metrics and experiments was essential.

WHAT CANDIDATES WISH THEY HAD PREPARED
Metrics fluency and structured-but-natural product reasoning. Candidates who
memorized frameworks but couldn't apply them to the company's actual product
struggled, as did those uncomfortable with A/B testing concepts.

OUTCOME: Offer made. A strong, structured metrics-diagnosis answer and genuine
user empathy were the differentiators.
""",

    "interview_fresher_ml_midsize.txt": """\
INTERVIEW EXPERIENCE: Fresher ML role at a mid-size company

ROUNDS & STRUCTURE
Three rounds. (1) Online test: aptitude, coding, ML MCQs. (2) Technical interview:
ML fundamentals, a coding problem, and projects discussion. (3) HR/managerial fit.

SPECIFIC QUESTIONS ASKED
- "Explain bias-variance tradeoff."
- "How does gradient descent work? What is a learning rate?"
- Coding: a basic array/string problem (easy-medium).
- "Walk me through your final-year/college ML project — why these choices?"
- "What happens if your training accuracy is high but test accuracy is low?"

WHAT THE INTERVIEWER EMPHASIZED
Fundamentals and depth on the candidate's own projects. For a fresher, they did NOT
expect production experience; they wanted clear understanding of basics and honest,
deep knowledge of whatever the candidate had actually done. Red flag: claiming a
project on the resume but not understanding it.

WHAT CANDIDATES WISH THEY HAD PREPARED
Truly understanding their own resume projects, and core ML math basics. Many
freshers listed projects they couldn't explain, which tanked the interview.

OUTCOME: Offer made. Solid fundamentals and the ability to deeply explain a
self-built project (over a flashy but poorly-understood one) won it.
""",

    "interview_senior_backend_us_india.txt": """\
INTERVIEW EXPERIENCE: Senior Backend Engineer at a US product company's India office

ROUNDS & STRUCTURE
Rigorous, five rounds. (1) Recruiter screen. (2) Two DS&A rounds (medium-hard,
real algorithmic depth). (3) System design round (senior-level, open-ended). (4)
Hiring manager / behavioral on ownership and cross-timezone collaboration. (5)
Bar-raiser.

SPECIFIC QUESTIONS ASKED
- DS&A: graph and dynamic-programming problems with optimal-complexity follow-ups.
- System design: "Design a globally distributed rate limiter" with deep discussion
  of consistency, failure, and scale.
- "How do you operate when the US team is offline and a decision is needed now?"
- "Tell me about a system you owned end to end, including on-call."

WHAT THE INTERVIEWER EMPHASIZED
A high, US-comparable bar: strong algorithms, rigorous system design with explicit
non-functional requirements, and the maturity to own decisions autonomously across
time zones. Excellent written/async communication was a recurring theme.

WHAT CANDIDATES WISH THEY HAD PREPARED
Hard DS&A even as a senior (some assumed seniority excused algorithm rounds — it
did not), and deep system design with reliability/consistency tradeoffs.

OUTCOME: Offer made. Strong system design and clear autonomy/ownership stories
carried it; the algorithm rounds were the gate, not the differentiator.
""",

    "interview_ai_researcher_deeptech.txt": """\
INTERVIEW EXPERIENCE: AI Researcher at a deep tech startup

ROUNDS & STRUCTURE
Five rounds. (1) Screen on research background and interests. (2) ML/math
fundamentals deep-dive. (3) Paper discussion: present your work or dissect a paper.
(4) Research coding exercise: reproduce/extend a result or design an experiment.
(5) Research-fit and culture with the founding researchers.

SPECIFIC QUESTIONS ASKED
- "Derive backprop for this layer" / probability and linear-algebra fundamentals.
- "Walk us through your most significant research contribution and its limitations."
- "Here's a paper — what's the key idea, what would you criticize, how would you
  extend it?"
- "Design an experiment to test hypothesis X; what are the confounders?"

WHAT THE INTERVIEWER EMPHASIZED
First-principles depth, research taste, and intellectual honesty about limitations.
They strongly valued the ability to read, reproduce, and critique papers, and to
reason about experimental rigor. API-only "users" of models were filtered out.

WHAT CANDIDATES WISH THEY HAD PREPARED
The math and the ability to defend research choices and admit limitations. Also
reproducing recent papers — candidates who only used libraries without
understanding internals struggled badly.

OUTCOME: Offer made. Demonstrated genuine depth (clean derivations, honest critique
of own work) and strong research taste were decisive.
""",

    "interview_rejected_candidate.txt": """\
INTERVIEW EXPERIENCE: A rejected candidate — what went wrong and why

CONTEXT
A mid-level backend engineer interviewing at a Series A product company. Strong
resume (good companies, relevant stack), but rejected after the onsite. This write-
up captures the failure modes interviewers cited, as a learning example.

ROUNDS & WHERE IT BROKE DOWN
(1) Screen — passed. (2) Coding — passed but slowly, with messy code. (3) System
design — this is where it failed. (4) Behavioral — mixed.

WHAT WENT WRONG
- System design: jumped to naming technologies ("we'll use Kafka, Redis, K8s")
  without clarifying requirements or estimating load. Could not justify WHY each
  piece was needed, and over-engineered a simple problem into premature
  microservices.
- Could not explain tradeoffs: when asked "why this database?", answered "it's what
  I've used" rather than reasoning about the workload.
- Behavioral: described work only as individual tasks; no ownership of outcomes, no
  examples of influencing others or handling ambiguity. Read as junior despite the
  years of experience.
- Communication: rambled, used jargon without structure, and didn't check
  assumptions with the interviewer.

WHAT THE CANDIDATE WISHED THEY HAD PREPARED
Driving an ambiguous system design with explicit requirements and tradeoffs, and
preparing concrete ownership/impact stories rather than task lists.

OUTCOME: No offer. The core reasons were weak system-design reasoning (tech name-
dropping over tradeoffs), lack of demonstrated ownership, and unclear
communication — not a lack of raw coding ability.
""",
}
