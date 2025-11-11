#!/usr/bin/env python3
"""
📄 ACL 2026 PAPER TEMPLATE
Auto-generate LaTeX submission
Your First Publication!
"""

PAPER_TEMPLATE = """
\\documentclass[11pt]{article}
\\usepackage{acl2026}

\\title{{Multi-Arc Resonance: A New Paradigm for Narrative NLG}}
\\author{{{name} \\\\ MIT CSAIL \\\\ {email}}}

\\begin{{document}}
\\maketitle

\\section{{Abstract}}
We introduce Chen's Multi-Arc Resonance model...

\\section{{Introduction}}
Narrative generation lacks emotional depth...

\\section{{Methodology}}
\\subsection{{Resonance Formula}}
$$R = \\sum w_i \\cdot T_i(t) \\cdot E_i(t)$$

\\section{{Experiments}}
Resonance Score: {resonance:.2f}

\\section{{Conclusion}}
This work advances computational creativity...
\\end{{document}}
"""


def generate_paper(name, email, resonance_score):
    paper = PAPER_TEMPLATE.format(name=name, email=email, resonance=resonance_score)
    with open("acl2026_submission.tex", "w") as f:
        f.write(paper)
    print("✅ ACL PAPER GENERATED: acl2026_submission.tex")


if __name__ == "__main__":
    from multi_arc_resonance import multi_arc_resonance

    resonance = multi_arc_resonance([[0, 3, 7, 10, 2]], ["Fear"])
    generate_paper("Your Name", "you@mit.edu", resonance)
