# UC-02: Batch Scoring
# Applies the trained model to all permission grants to generate the Least-Privilege Score.

print("Starting Batch Scoring...")

# 1. Load Model
# 2. Predict Probabilities (IsExcessive)
# 3. Aggregate to User Level -> Least-Privilege Score (0-100)

print("Scoring complete. Results saved to Lakehouse table 'UserScores'.")
