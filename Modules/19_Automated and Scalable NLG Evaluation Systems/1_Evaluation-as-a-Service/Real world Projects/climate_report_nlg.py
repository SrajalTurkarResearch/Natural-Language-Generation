# climate_report_nlg.py
# Real-World NLG: Scientific Climate Report
# Used by: IPCC, UNEP, NASA


def generate_climate_alert(data):
    """
    Generate urgent climate report
    """
    region = data["region"]
    temp_anomaly = data["temp_anomaly"]
    precipitation_change = data["precipitation_change"]
    risk_level = data["risk_level"]

    urgency = {"low": "Monitor", "medium": "Prepare", "high": "Act Now"}[risk_level]

    report = f"""
CLIMATE ALERT: {region.upper()} – {urgency}

OBSERVATIONS:
• Temperature anomaly: +{temp_anomaly}°C above average
• Precipitation: {precipitation_change:+}% change
• Risk Level: {risk_level.upper()}

RECOMMENDED ACTIONS:
"""
    if risk_level == "high":
        report += "• Immediate emission cuts\n• Activate disaster protocols"
    elif risk_level == "medium":
        report += "• Reduce water usage\n• Monitor infrastructure"
    else:
        report += "• Continue regular monitoring"

    report += (
        f"\n\nSource: Global Climate Model | {datetime.now().strftime('%Y-%m-%d')}"
    )
    return report


# === SAMPLE DATA ===
alert = {
    "region": "Mumbai",
    "temp_anomaly": 2.8,
    "precipitation_change": -45,
    "risk_level": "high",
}

if __name__ == "__main__":
    print(generate_climate_alert(alert))
