from jinja2 import Template
import random

# Function to scale opacity based on activation values (min 0.1, max 1.0)
def scale_opacity(activation, max_value):
    return max(0.1, min(activation / max_value, 1.0))

# Function to find the first valid concept (not null, not non-English)
def get_valid_concept(data):
    for concept in data['input_concept']:
        if concept and all(ord(c) < 128 for c in concept):  # Basic check for non-English characters
            return concept
    return "No valid concept"

# HTML template with dropdown for selecting ReAX ID and displaying the current concept
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tokenized Text Highlighting</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
        }
        h2 {
            text-align: center;
            color: #333;
        }
        h3 {
            text-align: center;
            color: #666;
        }
        table {
            width: 80%;
            margin: 0 auto;
            border-collapse: collapse;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            background-color: #fff;
            border-radius: 8px;
            overflow: hidden;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 15px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        .sae-highlight {
            background-color: rgba(255, 99, 132, {{ opacity }});
            font-size: 1em;
            padding: 2px 5px;
            border-radius: 5px;
        }
        .reax-highlight {
            background-color: rgba(54, 162, 235, {{ opacity }});
            font-size: 1em;
            padding: 2px 5px;
            border-radius: 5px;
        }
        .highlight:hover {
            cursor: pointer;
            border-bottom: 1px dotted black;
        }
        .dropdown {
            width: 200px;
            margin: 20px auto;
            text-align: center;
        }
        .concept {
            text-align: center;
            color: #444;
            font-weight: bold;
            margin: 20px auto;
        }
    </style>
    <script>
        function updateTable(reax_id) {
            const rows = document.querySelectorAll(".table-row");
            let concept = "";
            rows.forEach(row => {
                if (row.dataset.reaxId === reax_id) {
                    row.style.display = "";
                    if (!concept) {
                        concept = row.dataset.inputConcept;
                    }
                } else {
                    row.style.display = "none";
                }
            });
            document.getElementById("current-concept").innerText = concept ? concept : "No concept available";
        }
    </script>
</head>
<body>
    <h2>Tokenized Text with Highlights</h2>
    
    <div class="dropdown">
        <label for="reax-select">Select ReAX ID:</label>
        <select id="reax-select" onchange="updateTable(this.value)">
            {% for id in reax_ids %}
                <option value="{{ id }}" {% if id == default_reax_id %}selected{% endif %}>{{ id }}</option>
            {% endfor %}
        </select>
    </div>
    
    <div class="concept">
        <p>Current Concept: <span id="current-concept">{{ initial_concept }}</span></p>
    </div>

    <h3>SAE ID: <span id="sae-id">{{ sae_id }}</span></h3>
    <table>
        <thead>
            <tr>
                <th>Input Concept</th>
                <th>Category</th>
                <th>Tokenized Text (SAE)</th>
                <th>Tokenized Text (ReAX)</th>
            </tr>
        </thead>
        <tbody>
        {% for row in rows %}
            <tr class="table-row" data-reax-id="{{ row.reax_id }}" data-input-concept="{{ row.input_concept }}" {% if row.reax_id != default_reax_id %}style="display: none;"{% endif %}>
                <td>{{ row.input_concept }}</td>
                <td>{{ row.category }}</td>
                <td>{{ row.sae_text | safe }}</td>
                <td>{{ row.reax_text | safe }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

# Function to generate HTML content with dropdown for selecting ReAX ID and displaying the current concept
def generate_html_with_concept(data, tokenizer):
    rows = []
    sae_id = None
    
    # Get unique ReAX IDs for the dropdown
    reax_ids = sorted(data['reax_id'].unique())
    
    # Select a random ReAX ID for the default selection
    default_reax_id = random.choice(reax_ids)
    
    # Find the first valid concept
    initial_concept = get_valid_concept(data[data['reax_id'] == default_reax_id])
    
    # Iterate through the DataFrame rows to generate the initial table
    for _, row in data.iterrows():
        sae_id = row['sae_id']  # All rows will have the same sae_id in this case
        tokens = tokenizer.tokenize(row['input'])  # Tokenizer is fixed
        
        # Highlighting based on sae_acts and reax_acts with opacity scaling and raw values on hover
        sae_highlighted = [
            f'<span class="sae-highlight highlight" title="SAE Activation: {act:.3f}" style="background-color: rgba(255, 99, 132, {scale_opacity(act, row["max_sae_act"])});">{token}</span>'
            if act > 0 else token
            for token, act in zip(tokens, eval(row['sae_acts']))
        ]
        
        reax_highlighted = [
            f'<span class="reax-highlight highlight" title="ReAX Activation: {act:.3f}" style="background-color: rgba(54, 162, 235, {scale_opacity(act, row["max_reax_act"])});">{token}</span>'
            if act > 0 else token
            for token, act in zip(tokens, eval(row['reax_acts']))
        ]
        
        # Join highlighted tokens into a single string
        sae_text = ' '.join(sae_highlighted)
        reax_text = ' '.join(reax_highlighted)
        
        # Append the processed row
        rows.append({
            'input_concept': row['input_concept'],
            'category': row['category'],
            'sae_text': sae_text,
            'reax_text': reax_text,
            'reax_id': row['reax_id'],
        })
    
    # Render the HTML using Jinja2
    template = Template(html_template)
    html_content = template.render(rows=rows, sae_id=sae_id, reax_ids=reax_ids, initial_concept=initial_concept, default_reax_id=default_reax_id)
    return html_content