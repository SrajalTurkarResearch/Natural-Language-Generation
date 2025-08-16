# Case Study 1: OpenAI's GPT Models and Top-p (Nucleus) Sampling

- **Context:**  
  OpenAI's GPT-3 and subsequent models employ top-p (nucleus) sampling as the default decoding strategy for creative text generation tasks, such as storytelling.

- **Details:**  
  The 2019 paper _"Language Models are Unsupervised Multitask Learners"_ introduced top-p sampling to address the limitations of greedy and top-k methods. Top-p sampling dynamically selects the smallest set of high-probability tokens whose cumulative probability exceeds a threshold _p_, forming a "nucleus." This allows GPT-3 to generate outputs that are both diverse and coherent, reducing repetitive loops often seen with greedy decoding.

- **Impact:**  
  Top-p sampling led to significant improvements in human-evaluated story quality, establishing GPT-3 as a benchmark for creative natural language generation (NLG). For instance, when given the prompt "The astronaut landed on...", GPT-3 produces a wide range of plausible and imaginative story continuations.

- **Research Takeaway:**  
  For researchers, top-p sampling's adaptive nature is especially valuable for tasks demanding creativity, such as generating scientific hypotheses or brainstorming novel ideas in writing.
