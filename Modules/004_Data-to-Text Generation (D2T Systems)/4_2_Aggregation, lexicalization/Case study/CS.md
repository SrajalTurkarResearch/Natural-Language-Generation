# Case Studies: Aggregation and Lexicalization in NLG

## Case Study 1: Weather Report Generation

**System** : Arria NLG (used by weather services).
**Context** : Generates daily weather forecasts from meteorological data.
**Aggregation** : Combines temperature, sky condition, and wind speed into a single sentence.

- **Example** :
- **Input** : Temp = 20°C, Sky = Clear, Wind = 10 km/h.
- **Output** : “Expect a clear day with a high of 20°C and light winds at 10 km/h.”
- **How?** Uses syntactic coordination to merge related weather metrics.

  **Lexicalization** : Selects audience-friendly words like “clear” instead of “no clouds” and “light winds” instead of “10 km/h.”
  **Research Insight** : Balancing technical accuracy with simplicity is critical for public trust. Explore how aggregation rules affect user comprehension in weather apps.

## Case Study 2: Sports Summary Generation

**System** : Narrative Science’s Quill (used by ESPN).
**Context** : Generates summaries of sports matches.
**Aggregation** : Combines team scores, player stats, and match outcomes.

- **Example** :
- **Input** : Team A = 3 goals, Team B = 1 goal, Player X = 2 goals.
- **Output** : “Team A defeated Team B 3-1, with Player X scoring two goals.”
- **How?** Uses causal aggregation to link performance and outcome.

  **Lexicalization** : Chooses engaging verbs like “defeated” and player names for specificity.
  **Research Insight** : Study how lexicalization impacts fan engagement (e.g., vivid verbs vs. neutral terms).

## Case Study 3: Medical Report Generation

**System** : Custom NLG for hospital systems.
**Context** : Summarizes patient health data for doctors and patients.
**Aggregation** : Combines multiple health metrics into a concise summary.

- **Example** :
- **Input** : Blood Pressure = 140/90 mmHg, Heart Rate = 75 bpm, Condition = Hypertension.
- **Output** : “The patient has high blood pressure (140/90 mmHg) and a normal heart rate of 75 bpm, indicating hypertension.”
- **How?** Uses syntactic coordination and ellipsis to streamline metrics.

  **Lexicalization** : Uses “high” for patients and “hypertension” for doctors.
  **Research Insight** : Investigate how lexicalization affects patient trust and compliance in medical communication.
