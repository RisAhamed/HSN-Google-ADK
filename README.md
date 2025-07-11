﻿# HSN Code Validation and Suggestion AI Agent

## Project Overview

This project implements an intelligent agent using the **Google ADK (Agent Developer Kit) framework**. The agent's primary functions are to **validate Harmonized System Nomenclature (HSN) codes** and **suggest HSN codes based on product/service descriptions**. The validation and suggestions are performed against a master dataset provided in an Excel file (`HSN_data.xlsx`).


**Video Demonstration:**
*   [Link to Project Explanation Video - `AwesomeScreenshot-5_30_2025,4_59_05PM.mp4`](./AwesomeScreenshot-5_30_2025,4_59_05PM.mp4)
    *   *(Note: Ensure the video is in the repository and the link is relative, or host it and provide a direct URL).*

## Problem Domain: HSN Codes

HSN codes are an internationally standardized system of names and numbers used to classify traded products. They typically range from 2 to 8 digits, where each level of digits represents a more specific classification:
*   **01:** LIVE ANIMALS.
*   **0101:** LIVE HORSES, ASSES, MULES AND HINNIES.
*   **01011010:** A very specific type of horse.

This agent aims to assist users by:
1.  **Validating HSN Codes:** Checking if a given code is valid according to format, existence in master data, and hierarchical structure.
2.  **Suggesting HSN Codes:** Providing relevant HSN codes when a user describes a product or service.

## Key Features

*   **Comprehensive HSN Code Validation:**
    *   **Format Validation:** Checks if the code is numeric and adheres to standard lengths (2, 4, 6, 8 digits).
    *   **Existence Validation:** Verifies if the exact HSN code exists in the master dataset.
    *   **Hierarchical Validation:** Checks if parent codes of a given HSN code are present, ensuring structural integrity.
*   **Intelligent HSN Code Suggestion:**
    *   Accepts natural language descriptions of products/services.
    *   Utilizes **semantic search** (via sentence embeddings with `sentence-transformers`) to find the most relevant HSN codes.
    *   Returns a list of suggestions with HSN codes, descriptions, and similarity scores.
*   **ADK Framework Integration:**
    *   Built using `google.adk.agents.Agent`.
    *   Leverages a Large Language Model (Gemini Flash) for natural language understanding and tool orchestration.
    *   Employs Python functions as `tools` for deterministic logic.
*   **Efficient Data Handling:**
    *   Loads and pre-processes data from `HSN_data.xlsx` at startup for optimal query performance.
    *   Uses efficient data structures (sets for validation, pre-computed embeddings for suggestion).
*   **Interactive CLI:** Provides a command-line interface via `adk run` for user interaction.

## Technical Stack

*   **Framework:** Google ADK (Agent Developer Kit)
*   **Language:** Python
*   **Core Libraries:**
    *   `google-generativeai` (for LLM interaction via ADK)
    *   `pandas` (for data manipulation from Excel)
    *   `openpyxl` (Excel file reading engine for pandas)
    *   `sentence-transformers` (for generating text embeddings)
    *   `torch` (dependency for sentence-transformers)
    *   `python-dotenv` (for managing environment variables)

## Project Structure
```
HSN/
├── init.py # Initializes the HSN package, exposes the agent
├── hsn_final.py # Main Python script with agent logic (should be named hsn.py for adk run)
├── HSN_data.xlsx # Master HSN dataset (User must provide this)
├── readme.md # This README file
├── .env # to store the credentials, 
├── agent calling description tool.jpg # Screenshot: Agent using suggestion tool
├── description to information.jpg # Screenshot: Example suggestion output
├── image1.jpg # Screenshot: Example validation output
└── AwesomeScreenshot-5_30_2025,4_59_05PM.mp4 # Project demonstration video
```
*(Note: It's generally good practice to rename `hsn_final.py` to `hsn.py` if that's what `__init__.py` and `adk run HSN` expect, or update `__init__.py` accordingly.)*

## Setup and Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone https://github.com/RisAhamed/HSN-Agent
    # cd HSN
    ```

2.  **Create and Activate a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    google-adk
    google-generativeai
    python-dotenv
    pandas
    openpyxl
    sentence-transformers
    torch
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Key:**
    *   Create a file named `.env` in the `HSN/` directory.
    *   Add your Google API Key to it:
        ```
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        ```
    *   The application loads this key automatically.

5.  **Provide HSN Data File:**
    *   Place your HSN master data Excel file, named **`HSN_data.xlsx`**, inside the `HSN/` directory.
    *   The Excel file **must** contain at least two columns:
        *   `HSNCode` (Text/String format): The HSN code.
        *   `Description` (Text/String format): The description of the goods corresponding to the HSN code.

## How to Run the Agent

1.  Ensure your virtual environment is activated and you are in the parent directory of `HSN` (e.g., `C:/Users/username/`).
2.  Execute the following command in your terminal:
    ```bash
    adk run HSN
    ```
3.  The agent will start, and you can interact with it in the command line.

    **Example Queries:**
    *   `validate 01011010`
    *   `check code 9983`
    *   `what is the hsn for live horses`
    *   `suggest hsn for coffee beans, not roasted`
    *   `validate 1234` (test for non-existent code)
    *   `validate ABCDE` (test for invalid format)


## Screenshots

**HSN Code Validation Example:**
![Validation Example](./image1.jpg)

**HSN Code Suggestion (Tool Call & Output Examples):**
![Agent Calling Description Tool](./agent%20calling%20description%20tool.jpg)
![Description to Information](./description%20to%20information.jpg)


## Agent Design and Logic Deep Dive

### 1. Agent Architecture (Google ADK)
*   **Core Agent (`hsnagent`):** An instance of `google.adk.agents.Agent`.
*   **LLM:** Uses Google's Gemini Flash model (`gemini-2.5-flash-preview-05-20`) for natural language understanding, intent recognition, and response generation.
*   **Instruction Prompt:** A detailed system prompt guides the LLM's behavior:
    *   Differentiating between validation and suggestion requests.
    *   Extracting necessary entities (HSN code or product description).
    *   Selecting the appropriate tool.
    *   Formatting the final response based on structured tool output.
*   **Tools:** Python functions registered with the agent:
    *   `comprehensive_hsn_validation(code: str)`: Performs detailed validation of a given HSN code.
    *   `suggest_hsn_from_description(product_description: str, top_n: int = 3)`: Suggests HSN codes based on semantic similarity.

### 2. Data Handling
*   **Loading:** The `HSN_data.xlsx` is loaded into a pandas DataFrame at agent startup.
*   **Pre-processing & Efficiency:**
    *   **For Validation:**
        *   HSN codes are stored in a Python `set` (`hsn_codes_set`) for O(1) average time complexity for existence checks.
        *   A dictionary (`hsn_to_description`) maps HSN codes to their descriptions for quick retrieval.
    *   **For Suggestion:**
        *   Product descriptions are converted into numerical vector embeddings using `SentenceTransformer('all-MiniLM-L6-v2')` at startup.
        *   These pre-computed embeddings enable fast semantic similarity searches.
*   **Trade-off:** Pre-processing data at startup leads to a slightly longer initial load time but ensures significantly faster responses during user interaction, which is crucial for a good user experience.

### 3. Validation Logic (`comprehensive_hsn_validation`)
The tool performs a sequence of checks:
1.  **Format Validation (Numeric):** Ensures the code consists only of digits.
2.  **Format Validation (Length):** Checks if the code's length matches standard HSN lengths (e.g., 2, 4, 6, 8).
3.  **Existence Validation:** Verifies if the code exists in the `hsn_codes_set`.
4.  **Hierarchical Validation:** For valid codes longer than the base level (e.g., 2 digits), it checks if its parent codes (e.g., for `01011010`, it checks `010110`, `0101`, `01`) are also present in the `hsn_codes_set`.
The tool returns a structured dictionary detailing the outcome of each check.

### 4. Suggestion Logic (`suggest_hsn_from_description`)
1.  The input `product_description` is encoded into a query embedding using the pre-loaded `SentenceTransformer` model.
2.  **Cosine similarity** is computed between the query embedding and all pre-computed description embeddings from the master dataset.
3.  The HSN codes corresponding to the `top_n` (default 3) most similar descriptions are returned, along with their original descriptions and similarity scores.

### 5. Agent Response
*   **Valid HSN Code:** Confirmation, description, format status, and hierarchical status.
    *   Example: *"The HSN code 01011010 is valid. Description: [Its Description]. Format status is OK (Numeric, Length 8). Hierarchical status is OK (All parent codes exist)."*
*   **Invalid HSN Code:** Reason for invalidity (e.g., format error, not found, missing parent codes if critical).
    *   Example: *"The HSN code 010199 is invalid. Invalid format: HSN code length '6' is not standard. Expected lengths: {2, 4, 6, 8}. Validation summary: Invalid Format (Length)."*
*   **HSN Code Suggestions:** A list of suggested HSN codes, their descriptions, and similarity scores.
    *   Example: *"Here are some HSN code suggestions for 'live animals': 1. HSN: 01, Description: LIVE ANIMALS. (Similarity: 0.95)..."*


