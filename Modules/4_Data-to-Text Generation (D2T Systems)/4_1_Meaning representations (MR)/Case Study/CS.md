# Case Studies: Meaning Representations in NLG

These case studies illustrate how Meaning Representations (MRs) are applied in real-world NLG systems, providing concrete examples to support your learning and research.

## Case Study 1: Automated Weather Forecasting

**System** : BBC Weather App
**Context** : Generates daily weather reports for millions of users.
**MR** :

```
weather:
  - location: Tokyo
  - condition: sunny
  - temperature: 28°C
  - time: morning
```

**NLG Output** : “It’s sunny in Tokyo this morning with a temperature of 28°C.”
**How MR Works** : The MR captures core weather data (location, condition, temperature), enabling the system to generate reports in multiple languages (e.g., Japanese: “東京は今朝晴れで、気温は 28℃ です”) or styles (e.g., brief: “Sunny in Tokyo, 28°C”).
**Research Angle** : Investigate how MRs can incorporate real-time sensor data for hyper-localized forecasts.
**Note for Notebook** : Note the MR structure and its flexibility for multilingual outputs.

## Case Study 2: Sports Commentary Automation

**System** : Yahoo Sports Summaries
**Context** : Produces post-game summaries for football matches.
**MR (AMR)** :

```
(s / score-event
   :ARG0 (b / Barcelona)
   :ARG1 (r / Real-Madrid)
   :ARG2 (s2 / 2-1)
   :ARG3 (e / event :name football-match))
```

**NLG Output** : “Barcelona defeated Real Madrid 2-1 in an exciting football match.”
**How MR Works** : The AMR organizes game details (teams, score, event type), allowing varied outputs (e.g., “Barca won 2-1!”).
**Research Angle** : Explore how AMRs can capture dynamic events (e.g., goal times) for real-time commentary.
**Note for Notebook** : Sketch the AMR graph and list output variations.

## Case Study 3: Medical Report Generation

**System** : Hospital Report Generator
**Context** : Generates patient summaries from medical data.
**MR (Frame)** :

```
Diagnosis:
  - Patient: Jane
  - Condition: flu
  - Treatment: rest, fluids
```

**NLG Output** : “Jane has been diagnosed with the flu and is advised to rest and drink fluids.”
**How MR Works** : The frame structures medical facts, ensuring accurate and clear reports for doctors or patients.
**Research Angle** : Study how MRs can handle ambiguous medical terms (e.g., “fever” vs. specific diagnoses).
**Note for Notebook** : Write the MR and consider how to add more details (e.g., symptoms).

## Case Study 4: Virtual Assistant Responses

**System** : Alexa answering queries
**Context** : Responds to user questions like “What’s the capital of Brazil?”
**MR** :

```
capital-of:
  - country: Brazil
  - city: Brasília
```

**NLG Output** : “The capital of Brazil is Brasília.”
**How MR Works** : The MR captures the question’s intent, enabling concise and accurate responses.
**Research Angle** : Investigate MRs for handling complex queries (e.g., “Why is Brasília the capital?”).
**Note for Notebook** : Note the MR’s simplicity and brainstorm other question types.
