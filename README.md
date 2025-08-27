# SR TPM Appraisal Dashboard

**A local desktop dashboard to visualize EPA assessment data from Excel files, built with Streamlit.**  
**Automatically normalizes GM rotation scores for apples-to-apples comparison.**

***

## Features

- **Drag &amp; drop EPA Excel file:** No manual preprocessing needed.
- **GM normalization:** GM (General Medicine) scores automatically halved for fair comparison.
- **Resident breakdown:** Drill into results by resident.
- **Domain & trend charts:** Visualize scores across domains (PC, MK, SBP, PBLI, Prof, ICS, Overall) and time.
- **Averages:** See aggregate/average scores for quick comparisons.
- **Interactive and instant:** View tables, filter, and explore your data within seconds of uploading.

***

## Requirements

- **Python 3.8+** (works with Anaconda, Miniconda, or regular Python)
- **Recommended packages:**  
    - `streamlit`
    - `pandas`
    - `openpyxl`
    - `plotly`
    - (Make sure `protobuf` is version 3.20.x, see below if issue)

***

## Installation

1. Clone or download this repository, or save the Python file (e.g., `epa_dashboard.py`).
2. Install the dependencies (in Command Prompt/Terminal):

    ```bash
    pip install streamlit pandas openpyxl plotly protobuf==3.20.*
    ```

***

## Running the Dashboard

1. **Open your terminal or Anaconda Prompt.**
2. **Navigate to the folder** containing `epa_dashboard.py`.

    ```bash
    cd path/to/your/folder
    ```

3. **Start the Streamlit app:**

    ```bash
    streamlit run epa_dashboard.py
    ```

    - Your browser will open automatically to the dashboard.
    - Stop the server anytime with `Ctrl+C`.

***

## Usage

1. **Upload your EPA Excel file** (with a sheet named `'Quantitative'` that matches the provided sample's format).
2. **Explore resident results:** Select a resident, view scores, domain charts, and trends.
3. **See averages:** Quick comparisons of overall performance.
4. **Interact with the visualizations**—zoom, pan, or download charts as images.

***

## Troubleshooting

- **Protobuf error:**  
  If you see an error about `protobuf` versions or “Descriptors cannot be created directly,” run

    ```bash
    pip install protobuf==3.20.*
    ```

- Make sure your Excel matches the sample structure (sheet name and columns).

***

## Customization

- To add more visualizations or features, edit `epa_dashboard.py`.
- To support additional sheets or comment fields, adapt the parsing logic (pandas).

***

## License

*This dashboard is for internal/educational use. Adapt and share as you see fit!*

***

## Contact

For help, ideas, or suggestions, contact **[Your Name/Email/GitHub]**.

***

**Tip:**  
You can adapt this README if you later add new features, e.g., exporting, qualitative analysis, or web deployment.

Let me know if you want it tailored more to your context (e.g., hospital/department name, official logo, etc.)!

