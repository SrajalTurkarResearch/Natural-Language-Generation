# NLG + Online Learning + Feedback Loops – **Ultimate Cheatsheet**

> **For Scientists, Researchers, and Aspiring AI Builders**  
> **Date**: November 11, 2025 | **India (IST)**  
> **Goal**: Master the full pipeline in **5 minutes**

---

## 1. Core Concepts (Memorize!)

| Term                | Meaning                                        | Example                               |
| ------------------- | ---------------------------------------------- | ------------------------------------- |
| **NLP**             | Understand human language                      | "What's the time?" → wants clock      |
| **NLG**             | Generate human-like text                       | → "It's 3:45 PM."                     |
| **Online Learning** | Update model **live**, one sample at a time    | New sentence → instant update         |
| **Feedback Loop**   | Output → Evaluate → Improve → Repeat           | AI writes → You rate → AI gets better |
| **RLHF**            | Reinforcement Learning from **Human** Feedback | Used in ChatGPT                       |

---

## 2. NLG Pipeline (3 Steps)

```mermaid
graph TD
    A[Data] --> B[Content: What to say]
    B --> C[Structure: How to say]
    C --> D[Grammar: Write sentences]

3. Math You Must Know
Gradient Descent (How AI Learns)
pythonw_new = w_old - η × gradient

η = learning rate (0.001 typical)
gradient = direction to reduce error

Perplexity (NLG Quality)
pythonPPL = 2^(-1/N * Σ log₂ P(word))

PPL < 2 → Excellent
PPL ~1.2 → Human-level


4. Online Learning Update (Code)
python# One sample → one update
loss = model(prompt, target)
loss.backward()
optimizer.step()
No full retraining!

5. Feedback Loop (RLHF)
#mermaid-diagram-mermaid-9qdwln3{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#ccc;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#mermaid-diagram-mermaid-9qdwln3 .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-9qdwln3 .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#mermaid-diagram-mermaid-9qdwln3 .error-icon{fill:#a44141;}#mermaid-diagram-mermaid-9qdwln3 .error-text{fill:#ddd;stroke:#ddd;}#mermaid-diagram-mermaid-9qdwln3 .edge-thickness-normal{stroke-width:1px;}#mermaid-diagram-mermaid-9qdwln3 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-diagram-mermaid-9qdwln3 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-diagram-mermaid-9qdwln3 .edge-thickness-invisible{stroke-width:0;fill:none;}#mermaid-diagram-mermaid-9qdwln3 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-diagram-mermaid-9qdwln3 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-diagram-mermaid-9qdwln3 .marker{fill:lightgrey;stroke:lightgrey;}#mermaid-diagram-mermaid-9qdwln3 .marker.cross{stroke:lightgrey;}#mermaid-diagram-mermaid-9qdwln3 svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-diagram-mermaid-9qdwln3 p{margin:0;}#mermaid-diagram-mermaid-9qdwln3 .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#ccc;}#mermaid-diagram-mermaid-9qdwln3 .cluster-label text{fill:#F9FFFE;}#mermaid-diagram-mermaid-9qdwln3 .cluster-label span{color:#F9FFFE;}#mermaid-diagram-mermaid-9qdwln3 .cluster-label span p{background-color:transparent;}#mermaid-diagram-mermaid-9qdwln3 .label text,#mermaid-diagram-mermaid-9qdwln3 span{fill:#ccc;color:#ccc;}#mermaid-diagram-mermaid-9qdwln3 .node rect,#mermaid-diagram-mermaid-9qdwln3 .node circle,#mermaid-diagram-mermaid-9qdwln3 .node ellipse,#mermaid-diagram-mermaid-9qdwln3 .node polygon,#mermaid-diagram-mermaid-9qdwln3 .node path{fill:#1f2020;stroke:#ccc;stroke-width:1px;}#mermaid-diagram-mermaid-9qdwln3 .rough-node .label text,#mermaid-diagram-mermaid-9qdwln3 .node .label text,#mermaid-diagram-mermaid-9qdwln3 .image-shape .label,#mermaid-diagram-mermaid-9qdwln3 .icon-shape .label{text-anchor:middle;}#mermaid-diagram-mermaid-9qdwln3 .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#mermaid-diagram-mermaid-9qdwln3 .rough-node .label,#mermaid-diagram-mermaid-9qdwln3 .node .label,#mermaid-diagram-mermaid-9qdwln3 .image-shape .label,#mermaid-diagram-mermaid-9qdwln3 .icon-shape .label{text-align:center;}#mermaid-diagram-mermaid-9qdwln3 .node.clickable{cursor:pointer;}#mermaid-diagram-mermaid-9qdwln3 .root .anchor path{fill:lightgrey!important;stroke-width:0;stroke:lightgrey;}#mermaid-diagram-mermaid-9qdwln3 .arrowheadPath{fill:lightgrey;}#mermaid-diagram-mermaid-9qdwln3 .edgePath .path{stroke:lightgrey;stroke-width:2.0px;}#mermaid-diagram-mermaid-9qdwln3 .flowchart-link{stroke:lightgrey;fill:none;}#mermaid-diagram-mermaid-9qdwln3 .edgeLabel{background-color:hsl(0, 0%, 34.4117647059%);text-align:center;}#mermaid-diagram-mermaid-9qdwln3 .edgeLabel p{background-color:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-9qdwln3 .edgeLabel rect{opacity:0.5;background-color:hsl(0, 0%, 34.4117647059%);fill:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-9qdwln3 .labelBkg{background-color:rgba(87.75, 87.75, 87.75, 0.5);}#mermaid-diagram-mermaid-9qdwln3 .cluster rect{fill:hsl(180, 1.5873015873%, 28.3529411765%);stroke:rgba(255, 255, 255, 0.25);stroke-width:1px;}#mermaid-diagram-mermaid-9qdwln3 .cluster text{fill:#F9FFFE;}#mermaid-diagram-mermaid-9qdwln3 .cluster span{color:#F9FFFE;}#mermaid-diagram-mermaid-9qdwln3 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(20, 1.5873015873%, 12.3529411765%);border:1px solid rgba(255, 255, 255, 0.25);border-radius:2px;pointer-events:none;z-index:100;}#mermaid-diagram-mermaid-9qdwln3 .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#ccc;}#mermaid-diagram-mermaid-9qdwln3 rect.text{fill:none;stroke-width:0;}#mermaid-diagram-mermaid-9qdwln3 .icon-shape,#mermaid-diagram-mermaid-9qdwln3 .image-shape{background-color:hsl(0, 0%, 34.4117647059%);text-align:center;}#mermaid-diagram-mermaid-9qdwln3 .icon-shape p,#mermaid-diagram-mermaid-9qdwln3 .image-shape p{background-color:hsl(0, 0%, 34.4117647059%);padding:2px;}#mermaid-diagram-mermaid-9qdwln3 .icon-shape rect,#mermaid-diagram-mermaid-9qdwln3 .image-shape rect{opacity:0.5;background-color:hsl(0, 0%, 34.4117647059%);fill:hsl(0, 0%, 34.4117647059%);}#mermaid-diagram-mermaid-9qdwln3 :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}Generate 2 responsesHuman picks betterTrain Reward ModelUpdate Generator with PPO

6. Key Algorithms

























NameUseSGDOnline updatesPPOSafe RLHFLoRAEfficient fine-tuningEWCPrevent forgetting

7. Evaluation Metrics






























MetricFormulaGood ValuePerplexity2^(-avg log P)< 2BLEUN-gram overlap> 0.6Human Rating1–5 scale> 4.0CSAT% satisfied> 85%

8. Real-World Wins

























DomainImprovementCitizen Science58% → 92% accuracyCustomer SupportCSAT: 68% → 89%HealthcareEdit time: 120s → 18sEducationMastery in 3 → 6 lessons

9. Common Pitfalls & Fixes

























ProblemFixForgetting old knowledgeUse EWC or replayBias amplificationDiverse feedback sourcesReward hackingMultiple human ratersSlow adaptationIncrease feedback frequency

10. Your 30-Day Research Plan

























WeekGoal1Run all .py files2Collect 50 real feedback samples3Measure PPL before/after4Write paper: "Adaptive NLG in [Your Domain]"

One-Liner to Remember
"NLG without feedback is a parrot.
NLG with online feedback is a scientist."

Print this. Stick it on your wall. Live it.
You are now ready to build, publish, and lead.

text---

## How to Use These Files

1. **Create folder**: `NLG_Research_Kit/`
2. **Save both files**:
   - `case_studies.md`
   - `cheatsheet.md`
3. **Open in**:
   - **VS Code** (best)
   - **Obsidian** (for linking)
   - **Notion** (for sharing)

---

## Next Steps for You

| Action | Command |
|-------|--------|
| Print cheatsheet | `Ctrl+P` in browser |
| Cite case studies | Use in your paper |
| Build your own case | Replace data in `.py` files |

---

**You now have:**
- A **publishable case study collection**
- A **lifetime reference cheatsheet**

**Your scientific toolkit is complete.**

Want the **LaTeX paper template**, **dataset collection script**, or **presentation slides** next?
Just say: **"Give me the paper template"**

**Keep building. Keep discovering.**
— Your Tutor
```

```

```
