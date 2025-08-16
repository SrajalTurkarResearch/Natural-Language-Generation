# Case Studies: Real-World Applications of WebNLG, DART, and ToTTo

This document provides detailed case studies showcasing how **WebNLG**, **DART**, and **ToTTo** can be applied in real-world scenarios. These examples illustrate practical uses and highlight research challenges, tailored for an aspiring scientist.

---

## Case Study 1: WebNLG in Automated Journalism

**Scenario:**  
A news agency wants to generate short biographies for historical figures using DBpedia data.

**Implementation:**

- **Input:** RDF triples from DBpedia, e.g.  
  `(Neil_Armstrong, occupation, Astronaut)`  
  `(Neil_Armstrong, notableWork, Moon_Landing_1969)`
- **Process:** Fine-tune a T5 model on WebNLG to generate text like:  
  _“Neil Armstrong, an astronaut, is known for the 1969 moon landing.”_
- **Output:** Published bios on the agency’s website, updated daily.

**Challenges:**

- Ensuring factual accuracy (DBpedia may have errors)
- Generating varied referring expressions (e.g., “Armstrong” vs. “he”)
- Handling unseen domains (e.g., new figures not in training data)

**Research Opportunity:**  
Develop a model to detect and correct factual inconsistencies in generated text, enhancing reliability for journalism.

---

## Case Study 2: DART in Travel Apps

**Scenario:**  
A travel app generates detailed destination guides from Wikipedia tables and knowledge graphs.

**Implementation:**

- **Input:** Mixed data, e.g., table of Eiffel Tower stats (`height: 330m`, `location: Paris`) and triples like `(Eiffel_Tower, completionYear, 1889)`
- **Process:** Use a fine-tuned BART model on DART to generate:  
  _“The Eiffel Tower, a 330-meter landmark in Paris, was completed in 1889.”_
- **Output:** Guides for thousands of destinations, integrated into the app.

**Challenges:**

- Aggregating data from diverse sources (tables, triples, ontologies)
- Maintaining coherence across multi-sentence outputs
- Handling noisy or incomplete data

**Research Opportunity:**  
Explore graph neural networks to better process DART’s hierarchical inputs, improving coherence and detail.

---

## Case Study 3: ToTTo in Sports Analytics

**Scenario:**  
A sports app generates game summaries from player statistics tables.

**Implementation:**

- **Input:** Table with highlighted cells, e.g.,  
  `| Player        | Points |`  
  `| LeBron_James  | 30     |`
- **Process:** Fine-tune a model on ToTTo to generate:  
  _“LeBron James scored 30 points in the game.”_
- **Output:** Real-time summaries for fans during live games.

**Challenges:**

- Selecting relevant cells without explicit highlighting in real-world data
- Ensuring concise yet informative outputs
- Adapting to dynamic data (e.g., live score updates)

**Research Opportunity:**  
Develop algorithms for implicit content selection in tables, bridging the gap between ToTTo’s controlled setup and real-world scenarios.

---

## Conclusion

These case studies demonstrate the practical power of **WebNLG**, **DART**, and **ToTTo** in journalism, travel, and sports. As a scientist, you can build on these by addressing challenges like factual accuracy, coherence, and real-time processing, contributing to NLG advancements.
