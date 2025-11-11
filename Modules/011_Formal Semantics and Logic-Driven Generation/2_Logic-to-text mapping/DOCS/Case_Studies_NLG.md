# Case Studies: Logic-to-Text Mapping in Natural Language Generation (NLG)

This document presents four detailed case studies showcasing logic-to-text mapping in NLG across diverse fields: healthcare, sports, science, and education. Each case study is structured to provide context, input data, logical representations, output text, process details, lessons learned, challenges, and research prompts, inspiring you to apply NLG in your scientific career like Turing solving puzzles, Einstein simplifying complex ideas, or Tesla building practical solutions. These examples demonstrate real-world applications, highlight challenges, and spark ideas for your research.

## Case Study 1: Healthcare – Generating Patient Summaries

### Context

Hospitals use NLG to automate patient report generation from electronic medical records (EMRs), saving time and ensuring consistency. Logic-to-text mapping ensures accurate, human-readable summaries from structured data and logical rules.

### Input

- **Table** :

```
  Patient | Age | Symptoms        | Temperature
  John    | 70  | Fever, Cough   | 101°F
```

- **Logic** : Symptom(John, Fever) ∧ Symptom(John, Cough) ∧ Age(John, >65) → Recommend(DoctorVisit)

### Output

“John, aged 70, has a fever of 101°F and a cough, and should visit a doctor.”

### Process

1. **Parse Logic** : Verify the logic is valid (e.g., check ∧ and → syntax).
2. **Content Selection** : Choose key facts: fever, cough, age >65, and recommendation.
3. **Discourse Planning** : Structure as: state patient condition, then action.
4. **Lexicalization** : Map “Symptom” to “has,” “Recommend” to “should visit.”
5. **Aggregation** : Combine fever and cough into one sentence.
6. **Referring Expressions** : Use “John” once, then pronouns if repeated.
7. **Surface Realization** : Add grammar (e.g., “aged 70”).
8. **Evaluation** : Check for clarity and accuracy against EMR data.

### Lessons Learned

- **Impact** : Automating summaries reduces doctor documentation time by 30%, improving efficiency.
- **Accuracy Critical** : Missing symptoms (e.g., ignoring cough) could lead to incomplete advice.
- **Real-World Use** : Used in systems like Epic EMR for clinical notes.

### Challenges

- **Completeness** : Ensuring all relevant symptoms are included in the logic.
- **Ambiguity** : Logic must specify severity (e.g., “high fever” vs. “fever”).
- **Ethical** : Incorrect mappings could lead to wrong medical advice.

### Research Prompt

How could you extend the logic to handle multiple symptoms dynamically (e.g., fever, cough, and fatigue)? Could you add a rule to prioritize urgent symptoms?

## Case Study 2: Sports – Automated Game Recaps

### Context

Sports media outlets use NLG to generate engaging game recaps from statistical data, making content scalable for platforms like ESPN or Yahoo Sports. Logic-to-text ensures recaps are factual and exciting.

### Input

- **Table** :

```
  Player | Team  | Points | Assists
  LeBron | Lakers | 30     | 8
  Durant | Nets   | 25     | 5
```

- **Logic** : Points(LeBron, 30) ∧ Assists(LeBron, 8) ∧ Greater(Points(Lakers), Points(Nets))

### Output

“LeBron James scored 30 points and dished out 8 assists, leading the Lakers to a victory over the Nets, who were paced by Durant’s 25 points.”

### Process

1. **Parse Logic** : Confirm valid predicates and comparisons.
2. **Content Selection** : Focus on star player (LeBron), key stats, and game outcome.
3. **Discourse Planning** : Highlight individual performance, then team result.
4. **Lexicalization** : Map “Points” to “scored,” “Assists” to “dished out,” “Greater” to “victory.”
5. **Aggregation** : Combine points and assists into one clause.
6. **Referring Expressions** : Use “LeBron James” first, then “LeBron.”
7. **Surface Realization** : Add engaging verbs like “paced” for variety.
8. **Evaluation** : Use BLEU to compare with human-written recaps.

### Lessons Learned

- **Impact** : Enables real-time recaps for thousands of games, engaging fans.
- **Variety Needed** : Repetitive text (e.g., “scored X points” every game) bores readers.
- **Real-World Use** : Adopted by The Washington Post’s Heliograf for sports.

### Challenges

- **Diversity** : Generating varied sentences for similar stats.
- **Context** : Including intangibles like “team morale” not in logic.
- **Scalability** : Handling large datasets for entire leagues.

### Research Prompt

How could you add lexical variety (e.g., “racked up” vs. “scored”) automatically? Could you incorporate contextual logic (e.g., “clutch performance” for late-game points)?

## Case Study 3: Science – Climate Report Summaries

### Context

Climate researchers use NLG to communicate complex model outputs to policymakers and the public. Logic-to-text ensures precise, clear summaries from scientific data.

### Input

- **Table** :

```
  Year | GlobalTempRise | Region
  2025 | 1.2°C         | Global
```

- **Logic** : Increase(GlobalTemp, 1.2°C, 2025) ∧ Region(Global)

### Output

“In 2025, global temperatures have risen by 1.2 degrees Celsius compared to pre-industrial levels.”

### Process

1. **Parse Logic** : Validate the increase and region predicates.
2. **Content Selection** : Focus on temperature rise and year.
3. **Discourse Planning** : Present fact directly, then context.
4. **Lexicalization** : Map “Increase” to “have risen,” “Global” to “global.”
5. **Aggregation** : Single fact, no combination needed.
6. **Referring Expressions** : Use “global temperatures” as main subject.
7. **Surface Realization** : Add scientific precision (e.g., “Celsius”).
8. **Evaluation** : Verify accuracy against climate data standards.

### Lessons Learned

- **Impact** : Simplifies communication of critical data to non-experts.
- **Precision** : Errors in logic (e.g., wrong baseline) could mislead policy.
- **Real-World Use** : Used in IPCC report summaries.

### Challenges

- **Uncertainty** : Expressing model uncertainty (e.g., confidence intervals).
- **Complexity** : Summarizing multi-variable data (e.g., regional variations).
- **Trust** : Ensuring public trusts automated outputs.

### Research Prompt

How could NLG incorporate uncertainty (e.g., “likely 1.1–1.3°C”)? Could you map logic for regional climate impacts?

## Case Study 4: Education – Geometry Explanations

### Context

Educational platforms like Khan Academy use NLG to explain math concepts to students. Logic-to-text creates clear, tailored explanations from formal rules.

### Input

- **Logic** : Equals(Angle, 90°) → RightAngle(Triangle)
- **Context** : Teaching a middle school geometry lesson.

### Output

“The triangle has a right angle because one of its angles measures 90 degrees.”

### Process

1. **Parse Logic** : Check implication structure.
2. **Content Selection** : Focus on angle and conclusion.
3. **Discourse Planning** : Explain cause (90°) then result (right angle).
4. **Lexicalization** : Map “Equals” to “measures,” “RightAngle” to “has a right angle.”
5. **Aggregation** : Single logic, no combination.
6. **Referring Expressions** : Use “the triangle” as subject.
7. **Surface Realization** : Use simple, student-friendly words.
8. **Evaluation** : Test with students for clarity.

### Lessons Learned

- **Impact** : Makes learning interactive and scalable for online platforms.
- **Adaptability** : Explanations must match student age and knowledge.
- **Real-World Use** : Used in adaptive learning systems.

### Challenges

- **Personalization** : Adjusting for different learning levels.
- **Engagement** : Making explanations interesting, not robotic.
- **Correctness** : Ensuring logic covers all cases (e.g., non-right triangles).

### Research Prompt

How could NLG adapt explanations for different age groups (e.g., elementary vs. high school)? Could you add logic for interactive questions?

## Conclusion

These case studies show the power of logic-to-text mapping in NLG across healthcare, sports, science, and education. They highlight practical applications, challenges like bias and variety, and opportunities for research. Use these as inspiration for your own experiments, like Turing’s code-breaking, Einstein’s clear theories, or Tesla’s inventions.

**Research Lab** : Pick one case study. Design a small NLG project based on it (e.g., map new healthcare logic).
