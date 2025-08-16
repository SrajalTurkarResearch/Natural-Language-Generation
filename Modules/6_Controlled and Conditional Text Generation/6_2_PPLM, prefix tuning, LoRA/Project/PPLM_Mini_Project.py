# PPLM Major Project: Hiring Debias
resumes = ["Seeking strong leader"]


def debias(text):
    h_t = torch.tensor([0.5, -0.3, 0.2])
    h_new = attribute_gradient(h_t)
    debiased = text.replace("strong", "qualified")
    print(f"Debiased: {debiased} (h: {h_new})")


for r in resumes:
    debias(r)
