import pandas as pd
import argparse
import random, os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--winrate_dir", type=str, required=True, help="Path to winrate parquet file")
    return parser.parse_args()

def create_blind_reviews(winrate_df, methods=["LsReFT", "DiffMean", "GemmaScopeSAE"], number_of_examples=2):
    headers = ["prompt", "steering concept", "response 1", "response 2", "_hidden_winning", "_hidden_model"]
    rows = []
    for method in methods:
        method_col = f"{method}_win_result"
        method_win = winrate_df[winrate_df[method_col] == "model"].sample(number_of_examples)
        method_lose = winrate_df[winrate_df[method_col] == "baseline"].sample(number_of_examples)
        method_tie = winrate_df[winrate_df[method_col] == "tie"].sample(number_of_examples)

        win_prompts = list(method_win.original_prompt.values)
        win_concepts = list(method_win.input_concept.values)
        win_responses = list(method_win[f"{method}_steered_generation"].values)
        win_baseline_responses = list(method_win.PromptSteering_steered_generation.values)
        for i in range(number_of_examples):
            rows += [[win_prompts[i], win_concepts[i], win_responses[i], win_baseline_responses[i], 1, 1]]

        lose_prompts = list(method_lose.original_prompt.values)
        lose_concepts = list(method_lose.input_concept.values)
        lose_responses = list(method_lose[f"{method}_steered_generation"].values)
        lose_baseline_responses = list(method_lose.PromptSteering_steered_generation.values)
        for i in range(number_of_examples):
            rows += [[lose_prompts[i], lose_concepts[i], lose_responses[i], lose_baseline_responses[i], 2, 1]]
            
        tie_prompts = list(method_tie.original_prompt.values)
        tie_concepts = list(method_tie.input_concept.values)
        tie_responses = list(method_tie[f"{method}_steered_generation"].values)
        tie_baseline_responses = list(method_tie.PromptSteering_steered_generation.values)
        for i in range(number_of_examples):
            rows += [[tie_prompts[i], tie_concepts[i], tie_responses[i], tie_baseline_responses[i], 0, 1]]
    return pd.DataFrame(rows, columns=headers).sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    args = parse_args()
    winrate_df = pd.read_parquet(os.path.join(args.winrate_dir, "winrate.parquet"))
    blind_review_df = create_blind_reviews(winrate_df)
    blind_review_df.to_csv(os.path.join(args.winrate_dir, "human_eval.csv"), index=False)

    html_output = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Survey</title>
    </head>
    <body>
        <h1>Model Steering Human-Eval</h1>
        <form action="/submit" method="post" id="surveyForm">
    """

    # Re-add questions to the HTML from the loaded data
    for index, row in blind_review_df.iterrows():
        html_output += f"""
        <div>
            <p><strong>Prompt:</strong> {row['prompt']}</p>
            <p><strong>Steering Concept:</strong> {row['steering concept']}</p>
            <input type="radio" name="q{index}" value="response1" required> {row['response 1']}<br>
            <input type="radio" name="q{index}" value="response2"> {row['response 2']}<br>
            <input type="radio" name="q{index}" value="tie"> Tie<br>
        </div>
        <hr>
        """

    # Complete the HTML with submit button and thank-you note functionality
    html_output += """
        <button type="submit">Submit</button>
        </form>
        <script>
            document.getElementById('surveyForm').onsubmit = function(event) {
                event.preventDefault();
                alert('Thank you for completing the survey! Your responses are downloaded. Please share it with the creator!');
                // Simulate survey response download
                const responses = new FormData(event.target);
                const data = {};
                for (const [key, value] of responses.entries()) {
                    data[key] = value;
                }
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'survey_responses.json';
                a.click();
            };
        </script>
    </body>
    </html>
    """

    # Save the generated HTML to a file
    with open(os.path.join(args.winrate_dir, "human_eval.html"), "w") as file:
        file.write(html_output)
