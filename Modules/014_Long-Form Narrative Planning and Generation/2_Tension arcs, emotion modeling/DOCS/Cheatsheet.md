# Cheatsheet: Tension Arcs & Emotion Modeling in NLG

**Quick Reference Guide | For Aspiring Scientists**
Dr. Alex Chen | MIT CSAIL
_Use: Bookmark for coding, exams, interviews, or research planning. Fundamentals to advanced, in bullets._

---

## 1. Core Concepts

### Tension Arc

- **Definition:** Excitement curve in stories (low → high → low)
- **Stages:**
  - Exposition (0)
  - Rising (build)
  - Climax (peak)
  - Falling (resolve)
  - Resolution (end)
- **Analogy:** Like a roller coaster hill.
- **Why:** Keeps engagement; enables dynamic text in NLG.

### Emotion Modeling

- **Goal:** Generate text to evoke specific feelings (joy, fear, etc.)
- **Analogy:** Painting with words and sentences.
- **Pipeline:** Detection → Mapping → Lexicon → Syntax → Context

---

## 2. Emotion Frameworks

- **Plutchik Wheel:** 8 primaries (joy, sadness, fear, anger, anticipation, surprise, trust, disgust)
  - Intensities: Joy → Ecstatic
- **VAD Model:**
  - Valence (happy/sad: -1 to 1)
  - Arousal (calm/excited: 0–1)
  - Dominance (weak/strong: -1 to 1)
  - _Example:_ Fear = (-0.7, 0.9, -0.8)
- **Ekman Basics:** 6 universals (happy, sad, fear, angry, surprise, disgust)
- **Distance Formula:**
  $$
  d = \sqrt{(V_1-V_2)^2 + (A_1-A_2)^2 + (D_1-D_2)^2}
  $$

---

## 3. Math Formulas

- **Tension Function:**

  $$
  T(t) = \begin{cases}
    a t^2, & t < 0.5 \quad \text{(rising)} \\
    -a (t-1)^2 + 1, & t \geq 0.5 \quad \text{(falling)}
  \end{cases}
  $$

  - _a:_ Steepness (1–10)
  - _t:_ Progress (0–1)
  - _Example:_ a=4, t=0.3 → T=0.36

- **Multi-Arc Resonance (advanced):**

  $$
  R = \sum w_i T_i(t) E_i(t)
  $$

  - _w:_ weights (e.g., 1.0 plot, 0.6 romance)

---

## 4. NLG Implementation Steps

1. **Define Arc:** `[0, 3, 7, 10, 2]`
2. **Map Emotions:** Tension → VAD/Lexicon
   - _Lexicon:_
     - Curious: `['wondered']`
     - Terrified: `['screamed']`
3. **Style:**
   - Low tension: long sentences
   - High tension: short + "!"
4. **Generate:** Rule-based or Transformer (e.g., GPT-2)

- **Code Tip:** Use `EmotionNLG` class from `rule_based_nlg.py`

---

## 5. Tools & Libraries

- **Python:** `numpy`, `plotly` (visuals), `transformers` (GPT), `gradio` (apps)
- **Datasets:** EmoBank (emotion-labeled text)
- **Install:**
  ```
  pip install numpy plotly transformers gradio
  ```

---

## 6. Visualizations

- **Tension Plot:** Line graph (`x`: progress, `y`: 0–10)
- **VAD Space:** 3D scatter (axes: V, A, D)
- **Code:**
  - `plot_tension_arc(a=4)`
  - `visualize_vad_space()`

---

## 7. Real-World Tips

- **Games:** Sync arcs with events (e.g., COD: threat → panic)
- **Chatbots:** VAD for empathy (Headspace: anxiety → calm)
- **Ads/Education:** Short arcs for impact (Duolingo: streak → pride)
- **Metrics:** Retention (+%), ROI ($), NPS

---

## 8. Common Pitfalls & Fixes

- **Ambiguity:** Use VAD distance > 0.5 for emotion shifts
- **Bias:** Employ diverse lexicons, test for cultures
- **Over-Tension:** Cap values at 10; always resolve arcs

---

## 9. Exercises (Quick)

- **Basic:** Calc T(0.4, a=5) → 1.0
- **Intermediate:** VAD distance Fear-Happy ≈ 2.07
- **Advanced:** Generate a 3-sentence arc story

---

## 10. Research Roadmap

- **Year 1:** Master basics (this cheatsheet)
- **Year 2:** Build projects → ACL paper
- **Advanced:** Multi-arc resonance → NeurIPS
- **Ethics:** Avoid manipulation; ensure transparency

---

**Pro Tip:**
Run `master_runner.py` for all demos. Update with your experiments!

**Reference Files:**
.py projects, Jupyter notebook.

**Questions?**
Email: [grok@x.ai](mailto:grok@x.ai)

Ethics: Avoid manipulation; transparency.

Pro Tip: Run master_runner.py for all demos. Update with your experiments!
Reference Files: .py projects, Jupyter notebook. Email grok@x.ai for questions.
