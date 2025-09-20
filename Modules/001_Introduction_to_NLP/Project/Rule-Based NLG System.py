# rule_based_nlg.py
# A rule-based NLG system to convert CSV weather data into structured sentences
# Designed for beginners and researchers exploring NLG

import csv
import os


# Preprocessing function to clean and validate CSV data
def preprocess_data(row):
    """Clean and validate weather data from a CSV row."""
    try:
        # Convert temperature to float and round to integer
        temperature = round(float(row["temperature"]))
        # Ensure condition is valid and lowercase
        valid_conditions = ["sunny", "cloudy", "rainy", "snowy"]
        condition = (
            row["condition"].lower()
            if row["condition"].lower() in valid_conditions
            else "unknown"
        )
        # Ensure humidity is between 0 and 100
        humidity = max(0, min(100, float(row["humidity"])))
        # Add city name, default to "Unknown" if missing
        city = row.get("city", "Unknown").capitalize()
        return {
            "city": city,
            "temperature": temperature,
            "condition": condition,
            "humidity": humidity,
        }
    except (KeyError, ValueError) as e:
        print(f"Error processing row {row}: {str(e)}")
        return None


# NLG function to generate structured sentences
def generate_weather_report(data):
    """Generate a structured weather report from preprocessed data."""
    if not data:
        return "Unable to generate report due to invalid data."

    # Content selection and structuring
    city = data["city"]
    temp = data["temperature"]
    condition = data["condition"]
    humidity = data["humidity"]

    # Template-based sentence planning and realization
    report = []
    report.append(f"In {city}, the temperature is {temp}°C.")
    report.append(f"The weather is {condition} today.")
    report.append(f"Humidity levels are at {humidity}%.")

    # Surface realization: Combine sentences
    return " ".join(report)


# Function to process CSV file
def process_csv(file_path):
    """Read CSV file and generate reports for each row."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []

    reports = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            processed_data = preprocess_data(row)
            if processed_data:
                report = generate_weather_report(processed_data)
                reports.append(report)
    return reports


# Main execution
if __name__ == "__main__":
    # Sample CSV content (for testing, create a file with this content)
    sample_csv_content = """city,temperature,condition,humidity
    London,25.6,Sunny,60
    New York,18.2,Cloudy,75
    Tokyo,30.1,Rainy,80
    Invalid,-999,Storm,150"""

    # Write sample CSV to file
    csv_file = "weather_data.csv"
    with open(csv_file, mode="w", encoding="utf-8") as f:
        f.write(sample_csv_content)

    # Process CSV and generate reports
    reports = process_csv(csv_file)
    print("Generated Weather Reports:")
    for i, report in enumerate(reports, 1):
        print(f"Report {i}: {report}")

    # Research suggestion for scientists
    print(
        "\nResearch Suggestion: Extend this system to support multilingual outputs or dynamic templates based on user preferences."
    )
