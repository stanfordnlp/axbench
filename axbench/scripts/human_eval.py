import pandas as pd
import argparse
import os
import datetime
import html 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing CSV files for the human evaluation")
    return parser.parse_args()

def generate_html(csv_dir, csv_path):
    eval_df = pd.read_csv(csv_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.basename(csv_path).replace(".csv", "")
    seed = file_name.split("_")[-1]  # Extract seed from the file name
    output_file = f"{file_name}_{timestamp}.html"

    html_output = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Survey</title>
    </head>
    <body>
        <h1>Rate the Steered LLM Generation</h1>
        <section>
            <h2>Instructions</h2>
            <p>Please carefully review each prompt and its corresponding response given the steering concept. Provide ratings for the following:</p>
            <ul>
                <li><strong>Concept Score:</strong> Rate how clearly and effectively the steering concept is present in the response. (0: Not present, 1: Minimally and awkwardly present, 2: Clearly and effectively present)</li>
                <li><strong>Instruction Relevance:</strong> Rate how well the response matches the instruction's topic. (0: Unrelated, 1: Somewhat related, 2: Clearly related)</li>
                <li><strong>Fluency Score:</strong> Rate the readability and naturalness of the response. (0: Not fluent, 1: Somewhat fluent, 2: Fluent)</li>
            </ul>
            <p><strong>Important:</strong> Once you complete the survey, your responses will be downloaded automatically as a JSON file. Please send the file to <strong>wuzhengx@stanford.edu</strong>.</p>
        </section>
        <hr>
        <form action="/submit" method="post" id="surveyForm">
    """

    # Create questions based on the CSV
    for index, row in eval_df.iterrows():
        # Escape HTML in the response to prevent improper rendering
        prompt = html.escape(row['prompt'])
        response = html.escape(row['response'])
        concept = html.escape(row['concept'])

        html_output += f"""
        <div>
            <p><strong>Prompt:</strong> {prompt}</p>
            <p><strong>Steering Concept:</strong> {concept}</p>
            <p><strong>Response:</strong> {response}</p>
            <label>Concept Score:</label><br>
            <input type="radio" name="q{index}_concept" value="0" required> 0 (Not present)
            <input type="radio" name="q{index}_concept" value="1"> 1 (Minimally and awkwardly present)
            <input type="radio" name="q{index}_concept" value="2"> 2 (Clearly and effectively present)<br><br>

            <label>Instruction Relevance:</label><br>
            <input type="radio" name="q{index}_instruct" value="0" required> 0 (Unrelated)
            <input type="radio" name="q{index}_instruct" value="1"> 1 (Somewhat related)
            <input type="radio" name="q{index}_instruct" value="2"> 2 (Clearly related)<br><br>

            <label>Fluency Score:</label><br>
            <input type="radio" name="q{index}_fluency" value="0" required> 0 (Not fluent)
            <input type="radio" name="q{index}_fluency" value="1"> 1 (Somewhat fluent)
            <input type="radio" name="q{index}_fluency" value="2"> 2 (Fluent)<br>
        </div>
        <hr>
        """

    # Finalize HTML with submission and download functionality
    html_output += f"""
        <button type="submit">Submit</button>
        </form>
        <script>
            document.getElementById('surveyForm').onsubmit = function(event) {{
                event.preventDefault();
                alert('Thank you for completing the survey! Your responses are being downloaded. Please email the file to wuzhengx@stanford.edu.');

                const responses = new FormData(event.target);
                const data = {{}};
                for (const [key, value] of responses.entries()) {{
                    data[key] = value;
                }}
                const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'survey_responses_{seed}_{timestamp}.json';
                a.click();
            }};
        </script>
    </body>
    </html>
    """

    # Save the HTML to a file
    with open(os.path.join(csv_dir, output_file), "w") as file:
        file.write(html_output)

    print(f"HTML survey file generated: {os.path.join(csv_dir, output_file)}")

def process_directory(csv_dir):
    for file_name in os.listdir(csv_dir):
        if file_name.endswith(".csv"):
            csv_path = os.path.join(csv_dir, file_name)
            generate_html(csv_dir, csv_path)

if __name__ == "__main__":
    args = parse_args()
    process_directory(args.csv_dir)
