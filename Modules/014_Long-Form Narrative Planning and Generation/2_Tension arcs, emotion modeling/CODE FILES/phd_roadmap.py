#!/usr/bin/env python3
"""
🎓 PHD ROADMAP GENERATOR
Year-by-Year Plan to Tenure Track
"""

ROADMAP = {
    1: {"Milestone": "Master Basics", "Deliverable": "This notebook", "Journal": "-"},
    2: {
        "Milestone": "First Paper",
        "Deliverable": "Hybrid Tension Model",
        "Journal": "ACL Student",
    },
    3: {
        "Milestone": "MS Thesis",
        "Deliverable": "Multi-Modal Engine",
        "Journal": "EMNLP",
    },
    4: {
        "Milestone": "PhD Apps",
        "Deliverable": "2 Publications",
        "Journal": "Top-5 NLP",
    },
    5: {"Milestone": "PhD Year 1", "Deliverable": "Novel Theory", "Journal": "NeurIPS"},
    8: {
        "Milestone": "Faculty",
        "Deliverable": "Lab + Students",
        "Journal": "Tenure MIT",
    },
}


def print_roadmap():
    print("| Year | Milestone | Deliverable | Journal |")
    print("|------|-----------|-------------|---------|")
    for year, data in ROADMAP.items():
        print(
            f"| {year} | {data['Milestone']} | {data['Deliverable']} | {data['Journal']} |"
        )


if __name__ == "__main__":
    print_roadmap()
