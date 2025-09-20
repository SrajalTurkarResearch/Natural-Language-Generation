# Prefix Tuning Major Project: Educational Explainer
topics = ["Gravity is force"]


def explain(topic, level="simple"):
    exp = topic.replace("force", "pull") if level == "simple" else topic
    print(exp)


for t in topics:
    explain(t)
