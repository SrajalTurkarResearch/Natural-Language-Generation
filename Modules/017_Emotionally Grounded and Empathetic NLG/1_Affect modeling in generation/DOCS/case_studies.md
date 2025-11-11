# Case Studies in Affect Modeling for Natural Language Generation (NLG)

> **Real-World Applications | Technical Breakdowns | Impact Metrics | Research Opportunities** > _For Aspiring Scientists, Researchers, and AI Engineers_ > **Updated:** October 29, 2025

---

## Case Study 1: Woebot – Mental Health Chatbot with Empathetic NLG

| **Domain**       | Mental Health & Therapeutic AI                                |
| ---------------- | ------------------------------------------------------------- |
| **Company**      | Woebot Health (Stanford spin-off)                             |
| **Launch**       | 2017                                                          |
| **Core Tech**    | Rule-based + ML sentiment + GPT-style generation (fine-tuned) |
| **Affect Model** | Valence-Arousal + Ekman’s 6 emotions                          |

### How It Works

1. **Input Analysis** : User says _"I’m feeling overwhelmed and sad."_

- - VADER → `compound = -0.68` → **negative**
  - Valence: `-0.7`, Arousal: `+0.5` → **anxiety + sadness**

2. **Prompt Engineering** :

```text
   Respond with empathy and CBT grounding: User is feeling overwhelmed and sad.
```

2. **Output** :

> “I hear you—it sounds really heavy right now. Let’s take one small breath together. Ready?”

### Impact (2020 Clinical Study)

- **22% reduction** in depression (PHQ-9 scores) after 2 weeks
- **78% user retention** vs 45% for non-affective bots
- **Source** : Fitzpatrick et al., _JMIR_ (2020)

### Research Opportunities

- Longitudinal affect adaptation using reinforcement learning
- Cross-cultural empathy calibration (e.g., collectivist vs individualist)
- Integration with wearable biosensors (HRV → real-time arousal)

---

## Case Study 2: Persado – Emotion-Driven Marketing Copy

| **Domain**    | Marketing & Advertising                                   |
| ------------- | --------------------------------------------------------- |
| **Company**   | Persado (NYC)                                             |
| **Clients**   | JPMorgan, Verizon, Dell                                   |
| **Core Tech** | Affective Lexicon + Transformer fine-tuning + A/B testing |

### How It Works

1. **Emotion Target** : “Urgency + Trust”
2. **Lexicon Mapping** :

- Urgency: “now”, “limited”, “last chance”
- Trust: “secure”, “proven”, “guaranteed”

1. **Generated Variants** :

- “Secure your spot before it’s gone — limited seats!”
- “Act now — trusted by 10M+ users”

### Impact

- **+46% CTR** on emotional vs neutral subject lines
- **+19% conversion** in email campaigns
- **Source** : Persado Internal Benchmarks (2023)

### Research Opportunities

- Causal inference: Which emotion drives action per demographic?
- Bias audit: Are certain emotions overused for gender/race?
- Multimodal: Pair text with color psychology

---

## Case Study 3: AI Dungeon – Affective Storytelling Engine

| **Domain**     | Interactive Fiction & Gaming            |
| -------------- | --------------------------------------- |
| **Platform**   | AI Dungeon (Latitude)                   |
| **Core Model** | GPT-3 fine-tuned on story-emotion pairs |

### How It Works

- User sets tone: `dark`, `whimsical`, `romantic`
- Model conditioned via prompt:
  ```text
  [Tone: dark, suspenseful] The door creaks open slowly...
  ```
- Output adapts syntax, lexicon, pacing:
  > “ _A chill crawls up your spine as shadows twist like living smoke..._ ”

### Impact

- **4M+ users** , 100M+ stories generated
- **Emotion-aligned stories** rated 37% more immersive
- **Source** : Latitude User Surveys (2024)

### Research Opportunities

- Narrative coherence under emotional constraint
- User emotion induction via text (measure via GSR/fNIRS)
- Ethical boundaries: trauma triggers in generated horror

---

## Case Study 4: GovBot UK – Empathetic Public Service NLG

| **Domain**   | Government & Civic Tech             |
| ------------ | ----------------------------------- |
| **Agency**   | UK Government Digital Service (GDS) |
| **Use Case** | Benefits, Tax, Immigration Queries  |

### How It Works

- User: “I’m scared I’ll lose my home if I miss a payment.”
- System:
  1. Detects **fear + urgency**
  2. Uses **empathy template** + **policy facts**
  3. Output:
     > “I understand this is really worrying. You won’t lose your home immediately. Here’s a 3-step plan to get support…”

### Impact

- **+63% user satisfaction** vs robotic responses
- **40% fewer escalations** to human agents

### Research Opportunities

- Fairness across dialects (British vs regional English)
- Legal safety in affective government communication
- Privacy-preserving emotion logging

---

## Summary Table

| Case       | Domain         | Affect Model        | Key Metric        | Research Gap     |
| ---------- | -------------- | ------------------- | ----------------- | ---------------- |
| Woebot     | Mental Health  | Valence-Arousal     | 22% depression    | Biosensor fusion |
| Persado    | Marketing      | Lexicon + LLM       | +46% CTR          | Causal emotion   |
| AI Dungeon | Storytelling   | Prompt conditioning | +37% immersion    | Ethics in horror |
| GovBot UK  | Public Service | Template + Empathy  | +63% satisfaction | Dialect fairness |

---

> **Pro Tip for Scientists** :
> Replicate one case study using open datasets (e.g., EmoBank, DailyDialog). Publish a comparative analysis — instant conference paper!
