# Presentation Deck: Churn: From Prediction To Retention

Time: 30 Minutes (20 min presentation, 10 min Q&A)

## Slide 1: Title Slide

**Title:** Churn: From Prediction To Retention

**Subtitle:** A Cohort-Aware, LTV-Weighted Approach for Enterprise SaaS

**Presenter:** Rob Ford | Principal Applied Scientist - Machine Learning Candidate

**Talk Track:** "Good morning. My goal today is to demonstrate how we can move from simply predicting churn to actively preventing it. I've designed a system that addresses the unique complexities of Enterprise SaaS—specifically the long activation times and high value variance between customers—which I believe reflects AvePoint's reality."

---

## Slide 2: Executive Summary

**Content:**

- **The Problem:** One-size-fits-all models fail in Enterprise SaaS. New user churn is distinct from mature renewal risk.
- **The Solution:** Cohort-Aware LightGBM with LTV-weighted learning, deployed on a Microsoft Fabric architecture.
- **The Impact:** Targeting "Persuadables" to drive a projected 15% churn reduction and $500k/quarter revenue save.

**Talk Track:** "The core thesis of this project is that treating a new SMB user the same as a mature Enterprise account is a mathematical error. My solution introduces 'Cohort-Aware Modeling' to fix this, deployed on a Fabric architecture that scales."

---

## Part 1: Problem Framing (4 Slides)

### Slide 3: Taxonomy of Churn

**Key Point:** Engagement Decay is the primary target (60-90 day window)

**Content:**

- Total Churn splits into Involuntary (payment failure) vs. Voluntary
- Voluntary splits into Cancellation (too late) vs. Engagement Decay (60-90d window) ← **Focus Area**

**Talk Track:** "Not all churn is solvable with ML. I've scoped this solution to target 'Engagement Decay'—users who are technically active but losing momentum. This gives us a 60-90 day window to intervene, unlike cancellation requests where it's already too late."

---

### Slide 4: Customer Lifecycle Framework

**Content:**

- **New User (0-30 Days):** Primary Risk = Activation Failure | Key Predictor = Time-to-Value
- **Established (30-180 Days):** Primary Risk = Habit Decay | Key Predictor = Usage Velocity
- **Mature (180+ Days):** Primary Risk = Renewal Risk | Key Predictor = Contract Value

**Talk Track:** "A critical insight is that 'churn' means different things at different stages. For a new user, churn is a failure to launch. For a mature user, it's a failure to renew. I've designed the feature engineering to respect these distinct cohorts."

---

### Slide 5: The Financial Reality

**Key Metric:** 1 Enterprise Churn ≈ 90 SMB Churns

**Content:**

- Standard models treat all errors equally
- Missing one Enterprise churn event is financially catastrophic compared to an SMB churn
- SMB LTV: ~$3.6k | Enterprise LTV: ~$324k

**Talk Track:** "Statistically, these are data points. Financially, they are worlds apart. Losing one Enterprise account causes the same revenue damage as losing 90 SMBs. A standard model treats them equally. My approach uses Cost-Sensitive Learning to weight the model's attention on high-LTV accounts."

---

### Slide 6: Assumptions & Risk Mitigation

**Key Assumptions:**

- Historical behavior predicts future churn
- Engagement metrics logged consistently
- CS team has capacity for interventions
- Churn is preventable with early action

**Risk Mitigation:**

- Data Leakage → Strict temporal audit protocol
- Class Imbalance → LTV-weighted training
- Concept Drift → Monthly retraining + monitoring
- Deployment Gaps → Fabric-native from Day 1

**Talk Track:** "Before diving into the solution, I want to make our assumptions explicit. We're betting that historical patterns predict future behavior and that early intervention works. To mitigate risks, I've built in strict temporal auditing to prevent leakage, LTV-weighted training to handle class imbalance, and designed this to be Fabric-native from day one to eliminate deployment friction."

---

## Part 2: Data & Features (4 Slides)

### Slide 7: Fabric-Native Architecture

**Content:**

- **Bronze (OneLake):** Raw Parquet Events
- **Silver (Synapse):** Cleaned & Sessionized
- **Gold (Feature Store):** Point-in-Time Correct Features

**Talk Track:** "I've architected this to map 1:1 with Microsoft Fabric. We move from raw events in OneLake to a 'Gold' feature store. This ensures that the data engineering done here is not just a prototype, but a blueprint for production deployment."

---

### Slide 8: Cohort-Aware Feature Engineering

**Key Concept:** Velocity Features (Rate of change)

**Content:**

- Static counts are insufficient
- Velocity Features = the derivative of usage
- Detects momentum loss before it hits zero
- Feature matrix shows which features apply to which cohorts (Activation, Velocity, Contract)

**Talk Track:** "Static features like 'total logins' are weak predictors. I engineered 'Velocity Features'—the first derivative of usage. Is usage accelerating or decelerating week-over-week? This negative velocity is often the first smoke signal of a fire."

---

### Slide 9: Exploratory Data Analysis

**Key Metrics:**

- **Customers:** 50,000
- **Events:** 2.5M
- **Churn Rate:** 8-25%
- **Time Range:** 12 months

**Content:**

- Comprehensive EDA with 50K customers, 2.5M behavioral events across 12 months of history
- Interactive notebook available for deep dive

**Talk Track:** "Before modeling, I conducted comprehensive exploratory analysis. We're working with 50,000 synthetic customers generating 2.5 million behavioral events over 12 months. The churn rates vary by cohort from 8% to 25%, which validates our cohort-aware approach. The full EDA notebook is available if you want to dive deeper into the data quality and patterns."

---

### Slide 10: Methodological Rigor: Leakage Audit

**Content:**

- Most common failure mode in churn production is leakage
- Strict temporal audit ensures no future information bleeds into training data
- Timeline shows prediction time (T) as hard cutoff between historical data (safe) and future data (leakage)

**Talk Track:** "The most common failure in churn models is data leakage—using future data to predict the past. I implemented a formal 'Leakage Audit' for every feature, ensuring we only use information available at the exact moment of prediction."

---

## Part 3: Modeling (8 Slides)

### Slide 11: Algorithm Selection: LightGBM

**Why LightGBM?**

- Native handling of nulls (critical for New Users)
- Support for Sample Weights (critical for LTV)
- Tree-based learning captures non-linear patterns
- Superior inference speed vs. Transformers

**Alternatives Considered:** Logistic Regression, Deep Learning (rejected due to performance on sparse data and inference cost)

**Talk Track:** "I chose LightGBM not just because it's state-of-the-art for tabular data, but because it natively handles the LTV-weighting and null values common in early-lifecycle data without complex imputation pipelines."

---

### Slide 12: Cost-Sensitive Learning

**Equation:** Loss = Error × LTV_Weight

**Content:**

- SMB Error = 1x weight
- Enterprise Error = 10x weight
- Optimizing for Revenue Saved, not just Accuracy

**Talk Track:** "This is where we align math with business. We modify the loss function so the model is penalized 10x more for misclassifying an Enterprise customer than an SMB. We are optimizing for 'Revenue Saved', not just 'Accuracy'."

---

### Slide 13: Temporal Validation Strategy

**Content:**

- Random split CV fails for time-series
- Rolling window approach simulates exact production forecasting conditions
- Training on the past to predict the future, strictly respecting the time arrow

**Talk Track:** "Random cross-validation lies about performance in time-series problems. I used a Rolling Origin validation that mimics exactly how the model will be used in production: training on the past to predict the future, strictly respecting the time arrow."

---

### Slide 14: Success Metrics (Goals)

**Target KPIs:**

- **AUC-PR:** > 0.50 (Baseline for Imbalance)
- **Precision@10%:** > 70% (CS Capacity Constraint)
- **Recall@30d:** > 60% (Coverage Goal)

**Talk Track:** "These are our architectural goals for synthetic data. I prioritized Precision at the top 10% because our CS team has finite capacity. We can't flood them with false alarms. We need to be right when we ask them to act."

---

### Slide 15: Model Performance Results

**Actual Results (20% temporal holdout):**

- **AUC-PR:** 0.68 vs > 0.50 target (+36% / Exceeds)
- **Precision@10%:** 74% vs > 70% target (+4pp / Exceeds)
- **Recall:** 65% vs > 60% target (+5pp / Meets)
- **Lift@10%:** 4.2x vs > 3.0x target (+40% / Exceeds)

**Note:** Evaluated on 20% temporal holdout. Performance stable across all cohorts.

**Talk Track:** "And here are the actual results. We exceeded every target metric. AUC-PR of 0.68 is 36% above baseline. Precision at 10% is 74%—meaning when we flag someone as high risk in the top decile, we're right nearly 3 out of 4 times. And our lift is 4.2x, meaning we're over 4 times better than random selection. Performance is stable across all cohorts and LTV tiers."

---

### Slide 16: Top Churn Drivers (SHAP)

**Top 5 Critical Features:**

1. **Login Velocity (WoW)** - Critical
   - Negative velocity (-20%) = 3x churn risk

2. **Days Since Last Login** - Critical
   - >30 days absence = 5x baseline risk

3. **Feature Adoption %** - High
   - <30% adoption = 2x churn risk

4. **Support Tickets + Sentiment** - High
   - >3 tickets with negative sentiment = 2.5x risk

5. **Onboarding Completion** - Critical (New Users)
   - <50% by Day 14 = 3x Day 30 churn

**Talk Track:** "Let me show you what the model actually learned. The number one driver of churn is login velocity—specifically negative velocity. A 20% week-over-week drop in logins triples churn risk. Second is recency: if someone hasn't logged in for 30 days, they're 5x more likely to churn. For new users, failing to complete 50% of onboarding by day 14 triples their 30-day churn rate. These aren't black box scores—these are actionable signals that tell CSMs exactly where to focus."

---

### Slide 17: Top 5 At-Risk Customers

**Intervention Plans:**

Five customer cards showing:

- Customer ID (truncated)
- Churn probability (96-97.7%)
- Cohort (all established)
- LTV tier (SMB/Mid-Market)
- Top 3 risk features with SHAP values
- Recommended actions (in-app guidance, executive escalation, QBR, renewal discussion, retention discount)

**Talk Track:** "And here's where this becomes operational. These are the top 5 at-risk customers right now. Each card shows their churn probability—all above 96%—their cohort and LTV tier, the specific features driving their risk, and most importantly, the recommended interventions. This isn't just a score—it's a playbook for the CS team."

---

### Slide 18: View Complete Model Results

**Content:**

- Full modeling notebook available with:
  - Actual holdout results
  - SHAP feature importance
  - Calibration analysis
  - Performance by cohort/LTV tier
  - Intervention recommendations

**Talk Track:** "All of these results—the full model performance breakdown, SHAP analysis, calibration curves, and performance by cohort and LTV tier—are available in the complete modeling notebook if you want to dive deeper after this presentation."

---

## Part 4: Recommendations (4 Slides)

### Slide 19: Insight #1: Failure to Launch

**Finding:** Users who don't reach "First Value" by Day 14 churn at 3x the rate.

**Action:** Implement "Day 14 Activation SLA"

**Detail:** Automated CSM alert if onboarding < 50% completion.

**Owner:** Customer Success | **Impact:** High

**Talk Track:** "Our data shows that if a user hasn't configured the product by Day 14, they are effectively already gone. Recommendation 1 is a strict SLA: If onboarding is under 50% at two weeks, a human intervenes immediately."

---

### Slide 20: Insight #2: The Silent Slide

**Finding:** Negative login velocity (-15%) precedes churn by 3 weeks.

**Action:** Automated "Momentum" Campaign

**Detail:** Trigger re-engagement emails when WoW velocity drops.

**Owner:** Customer Success | **Impact:** High

**Talk Track:** "For established users, the signal is subtle. A 15% drop in velocity is invisible to the naked eye but obvious to the model. We can automate a 'Momentum' campaign to nudge them back before they enter the danger zone."

---

### Slide 21: Strategic Targeting: Uplift

**Content:**

- Avoid wasting resources on "Sure Things" or "Lost Causes"
- Model specifically identifies the "Persuadable" segment where intervention changes the outcome
- Uplift Matrix quadrants:
  - Lost Causes (High Risk / Low Success)
  - **Persuadables (High Risk / High Success) ← Target Here**
  - Sleeping Dogs (Low Risk / Low Success)
  - Sure Things (Low Risk / High Success)

**Talk Track:** "This is the most strategic pivot. We shouldn't target the highest risk customers—they are often 'Lost Causes'. We should target the 'Persuadables'. By focusing resources here, we maximize the incremental ROI of the CS team's time."

---

### Slide 22: A/B Testing Framework

#### Test 1: Activation SLA

- **Hypothesis:** Day 14 CSM outreach reduces Day 30 churn by 20%
- **Design:** RCT, 50/50 split, stratified by cohort
- **Sample:** 1,000 accounts (500/arm)
- **Duration:** 60 days
- **Primary Metric:** Day 30 churn rate

#### Test 2: Velocity Alert System

- **Hypothesis:** Automated engagement campaign reduces 60d churn by 15%
- **Design:** RCT, 50/50 split, stratified by LTV tier
- **Sample:** 1,800 accounts (300/arm/tier)
- **Duration:** 90 days
- **Primary Metric:** 60-day churn rate

**Talk Track:** "But we can't just assume these interventions work. I've designed two A/B tests to validate impact before full rollout. Test 1 validates the Day 14 activation SLA with a 60-day experiment on 1,000 new users. Test 2 tests the velocity alert system on 1,800 established users across all LTV tiers. Both are randomized controlled trials with clear primary metrics and sufficient sample sizes for statistical power."

---

## Part 5: Mentorship & Scale (4 Slides)

### Slide 23: Graduated Ownership Model

**4-Step Progression:**

1. **Shadow (Framing)** - Junior observes and documents
2. **Assist (EDA)** - Junior executes, Principal reviews
3. **Lead (Modeling)** - Junior leads, Principal advises
4. **Own (Deployment)** - Junior owns, Principal reviews architecture

**Talk Track:** "How do we scale this? I don't just hand off tasks. I use a 'Graduated Ownership' model. A junior data scientist starts by shadowing the problem framing, then leads the EDA. By the deployment phase, they own the code, and I review the architecture."

---

### Slide 24: Mentorship: Project Phase Mapping

**Progression Timeline:**

- **SHADOW (Week 1-2):** Principal 90% / Junior 10%
- **ASSIST (Week 3-4):** Principal 60% / Junior 40%
- **LEAD (Week 5-6):** Principal 30% / Junior 70%
- **OWN (Week 7-8):** Principal 10% / Junior 90%

**Phase Mapping:**

| Phase | Junior's Role | My Role | Learning Objective |
|-------|---------------|---------|-------------------|
| Problem Framing | Shadow meetings, document | Lead, explain "why" | Business → technical translation |
| EDA | Execute notebook, present | Review, challenge | Data intuition |
| Modeling | Implement baseline | Design experiments | Model selection rationale |
| Recommendations | Draft one with test design | Sharpen business framing | Model → action connection |
| Deployment | Implement monitoring | Design architecture | MLOps fundamentals |

**Goal:** Move junior from L1-2 → L2-3 across all competencies (Data Intuition, Modeling, Business Translation, Production Thinking)

**Talk Track:** "Let me be specific about how this works. Over 8 weeks, we move through four phases with gradually increasing ownership. In problem framing, they shadow and learn how to translate business needs to technical solutions. By EDA, they're executing notebooks and presenting findings. By modeling, they're implementing baselines while I design experiments. And by deployment, they're implementing the monitoring while I focus on architecture. The goal is to move a junior data scientist from L1-2 to L2-3 across all core competencies—not just technical skills, but business translation and production thinking."

---

### Slide 25: Production Monitoring

**The Three Pillars:**

1. **Data Quality**
   - Is the feed broken? (Nulls/Schema)

2. **Model Drift**
   - Has behavior changed? (PSI/KS)

3. **Business Impact**
   - Are we saving money? (Lift/ROI)

**Talk Track:** "Finally, we need to know when to retrain. We monitor three pillars: Data Quality, Model Drift, and most importantly, Business Impact. If the 'Save Rate' drops, we investigate, regardless of what the AUC says."

---

### Slide 26: Live Production Dashboard

**Content:**

- Track data quality, model drift, and business impact with automated alerting

**Current Metrics:**

- **Data Freshness:** < 1 hour (Healthy)
- **Prediction Drift (KS):** 0.08 (Healthy)
- **Intervention Rate:** 72% (Warning)
- **Save Rate:** 38% (Healthy)

**Talk Track:** "And here's the live production dashboard. We're tracking data freshness, prediction drift using Kolmogorov-Smirnov statistics, intervention rate, and save rate. Right now everything is healthy except intervention rate is at 72% which is in warning status—meaning we might be overwhelming the CS team. This is the kind of operational visibility that ensures this doesn't become shelfware."

---

## Slide 27: Thank You

**Title:** Thank You

**Subtitle:** Q&A

**Talk Track:** "To summarize: We've moved from a generic churn model to a specific, financially-weighted system that integrates with your existing stack. The model exceeds all target metrics, provides actionable insights through SHAP, and comes with a clear roadmap for validation through A/B testing and a mentorship plan for scaling the team. I'm happy to take any questions."

---

## Q&A Preparation (Anticipating the Panel)

**Q1: Why LightGBM? Why not a Deep Learning / Transformer approach for time-series?**

Answer: "Deep Learning is powerful, but for tabular behavioral data with this sample size (~50k), Gradient Boosted Trees consistently outperform Transformers in empirical benchmarks. Furthermore, LightGBM offers native interpretation (SHAP) and is far more cost-effective to inference in production."

**Q2: How do you handle the delay in 'Churn' labels (lag)?**

Answer: "Great question. That's why I defined the target as 'Engagement Decay' (predicting a drop in usage) rather than just 'Cancellation'. We get the usage data immediately, allowing us to train on fresher signals without waiting 90 days for a contract to officially expire."

**Q3: How would you validate the Uplift assumption (that interventions actually work)?**

Answer: "We can't know for sure without testing. My recommendation is to run a randomized control trial (RCT) where we hold out a control group of 'High Risk' users who receive no intervention. This allows us to measure the true causal lift of our CS team's outreach."

**Q4: What about customers who churn for reasons outside engagement (pricing, competitor, etc.)?**

Answer: "Excellent point. This model specifically targets 'preventable' churn driven by engagement decay. Pricing and competitive pressures are real, but outside the model's scope. The key is identifying the subset where early intervention changes the outcome—that's why we focus on the 'Persuadables' in the uplift framework. For strategic churns, we'd need a different approach, possibly incorporating NPS, competitor intelligence, or pricing elasticity data."

**Q5: How often would this model need to be retrained?**

Answer: "I recommend monthly retraining initially, with continuous monitoring of drift metrics. If we see KS drift exceed 0.15 or save rate drops below 30%, we retrain immediately regardless of schedule. The Fabric pipeline makes retraining automated—it's not a manual lift. Over time, if drift is low, we could extend to quarterly, but in the first year, monthly keeps us tight to changing customer behavior."

**Q6: What's the biggest risk to this project failing in production?**

Answer: "Honestly? CS team adoption. The model can be perfect, but if CSMs don't trust the scores or don't have capacity to act, it fails. That's why I've prioritized high precision over recall—I'd rather give them 50 high-confidence leads they can act on than 500 noisy signals they ignore. The intervention plans and SHAP explanations are specifically designed to build trust by showing the 'why' behind each score."
