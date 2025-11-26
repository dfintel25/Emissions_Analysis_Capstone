# Repository: Emissions Analysis, Livestock vs Vehicles
#### Developer: Derek Fintel
#### Contact: s542635@nwmissouri.edu

## Repository Overview
This repository captures the technical analysis utilized in a capstone project called "Emissions Analysis, Livestock vs Vehicles". The capstone project is part of a Masters of Science Data Analytics program.

#### Capstone Project Link: https://www.overleaf.com/read/rrpbnffjgngk#d6ef5d

#### Capstone Abstract
This study applied data science and machine learning techniques to compare and analyze greenhouse gas emissions from livestock production and vehicle usage. By conforming disparate datasets and performing regression and clustering analyses, the research quantified the relative emissions impact of cattle meat production and vehicle travel. The findings demonstrate that livestock-related emissions substantially exceed those from vehicle use under comparable annual activity assumptions providing insights into how diet and transportation choices may influence climate impact.

### Source Data:
This project utilized two separate datasets, sourced from [Kaggle](https://www.kaggle.com/). Both native CSV files have been brought into this project.

Livestock Dataset: https://www.kaggle.com/datasets/amandaroseknudsen/gleamlivestockemissions

Vehicle Dataset: https://www.kaggle.com/datasets/brsahan/vehicle-co2-emissions-dataset

### Use-case:
This project will emulate a business solution where a coffee shop's point of sale system's data outputs are fed into an automated streaming system that ingests sources data, produces and publishes topics of data, consumes and transforms the messages, and generates output files and visualizations.

### Visualization:
Our project utilizes a dyanmic and live streaming visualization tool called [StreamLit](https://streamlit.io/). This tool enables our consumer data to be processed & streamed through a Windows Subsystem for Linux (WSL) terminal.
Once the kafka system, Producer, and Consumer are all running, you can run the [live_sales_dashboard.py](https://vscode.dev/github/dfintel25/custom_pipeline_clean/blob/main/visualizations/live_sales_dashboard.py) program and it will prompt you to select an html viewer to access the streaming visualization.
![Link Selection Example](image.png)

Once activated, our visualization will show the following examples:


### Preliminary Setup Steps

### 1. Initialize
```
1. Click "New Repository"
    a. Generate name with no spaces
    b. Add a "README.md"
2. Clone Repository to machine via VS Code
    a. Create folder in "C:\Projects"
3. Install requirements.txt
4. Setup gitignore
5. Test example scripts in .venv
```
### 2. Create Project Virtual Environment
```
py -m venv .venv
.venv\Scripts\Activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
Retrieve installed items: !pip list
```
### 3. Git add, clone, and commit
```
git add .
git clone "urlexample.git"
git commit -m "add .gitignore, cmds to readme"
git push -u origin main
```
### 4. If copying a repository:
```
1. Click "Use this template" on this example repository (if it's not a template, click "Fork" instead).
2. Clone the repository to your machine:
   git clone example-repo-url
3. Open your new cloned repository in VS Code.
```
### 5. Detailed Project Setup
For additional setup details, see [SET_UP_Workflow.md](https://vscode.dev/github/dfintel25/Emissions_Analysis_Capstone/blob/main/SET_UP_Workflow.md)

```


