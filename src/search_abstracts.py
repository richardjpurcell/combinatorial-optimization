import pandas as pd

def search_csv_for_keywords(file_path, keywords, output_file=None):
    """
    Search a CSV file for abstracts containing specific keywords.

    :param file_path: Path to the CSV file to search.
    :param keywords: List of keywords to search for in the "Abstract" column.
    :param output_file: Path to save the filtered results as a new CSV file (optional).
    :return: Filtered DataFrame with matching rows.
    """
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Check if 'Abstract' column exists
        if 'Abstract' not in data.columns:
            raise ValueError("The CSV file does not contain an 'Abstract' column.")

        # Convert keywords to lowercase for case-insensitive matching
        keywords = [keyword.lower() for keyword in keywords]

        # Filter rows where the abstract contains any of the keywords
        filtered_data = data[data['Abstract'].str.contains('|'.join(keywords), case=False, na=False)]

        # Save the filtered results to a new file if output_file is specified
        if output_file:
            filtered_data.to_csv(output_file, index=False)
            print(f"Filtered results saved to {output_file}")

        return filtered_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Path to your CSV file
    file_path = "export2024.12.21-13.20.53.csv"

    # List of keywords to search for
    keywords = [
        "combinatorial optimization", "traveling salesman problem", "knapsack problem", "nurse scheduling problem",
        "multi-objective optimization", "dynamic programming", "resource allocation", "routing algorithms", "task scheduling",
        "mobile sensing platforms", "environmental data collection", "wildfire detection", "habitat conservation", "pollution tracking",
        "energy-efficient monitoring", "unmanned aerial vehicles", "drones in environmental monitoring", "ground robots",
        "hybrid monitoring systems", "drone path optimization", "sensor deployment", "remote sensing", "battery life",
        "payload capacity", "terrain constraints", "environmental constraints", "dynamic conditions", "heterogeneous platforms",
        "fire risk assessment", "emergency response optimization", "sensor network optimization", "wildland-urban interface",
        "case studies in wildfire monitoring"
    ]

    # Path to save filtered results (optional)
    output_file = "filtered_results.csv"

    # Search the CSV file
    results = search_csv_for_keywords(file_path, keywords, output_file)

    if results is not None:
        print("Filtered Data:")
        print(results.head())
