# project_legal_contract.py
# Real-World: Generate Contract Clauses with Compliance
# Use: Law Firms, LegalTech, Smart Contracts

from utils import neural_summary_nlg


def legal_clause_nlg(party_a, party_b, amount, deadline, penalty_rate=0.1):
    """
    Generate payment clause with legal safeguards
    """
    # === SYMBOLIC: Legal Rules ===
    clauses = [
        f"{party_a} shall pay {party_b} the amount of ${amount:,.2f} by {deadline}.",
        f"Late payment incurs penalty of {penalty_rate*100}% per month.",
        "Dispute resolution via arbitration in New York.",
        "This clause is binding under applicable law.",
    ]

    # === NEURAL: Rephrase for clarity ===
    plain_english = neural_summary_nlg(
        f"Payment terms: {party_a} pays {party_b} {amount} dollars by {deadline}. "
        f"Penalty applies if late. Legal in New York."
    )

    return (
        f"CONTRACT CLAUSE\n"
        + "=" * 50
        + "\n"
        + "\n".join(clauses)
        + "\n\n"
        + f"PLAIN ENGLISH:\n{plain_english}"
    )


# === TEST ===
if __name__ == "__main__":
    print(
        legal_clause_nlg(
            party_a="Acme Corp",
            party_b="Beta LLC",
            amount=50000,
            deadline="December 31, 2025",
        )
    )
