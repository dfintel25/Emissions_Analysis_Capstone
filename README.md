# Repository: Emissions Analysis, Livestock vs Vehicles
#### Developer: Derek Fintel
#### Contact: s542635@nwmissouri.edu

## Repository Overview
This repository captures the technical analysis utilized in a capstone project called "Emissions Analysis, Livestock vs Vehicles". The capstone project is part of a Masters of Science Data Analytics program and employs various data science and analytics methods against a problem domain.

#### Final Capstone Report Link: https://www.overleaf.com/read/rrpbnffjgngk#d6ef5d

### Capstone Abstract
*This study applied data science and machine learning techniques to compare and analyze greenhouse gas emissions from livestock production and vehicle usage. By conforming disparate datasets and performing regression and clustering analyses, the research quantified the relative emissions impact of cattle meat production and vehicle travel. The findings demonstrate that livestock-related emissions substantially exceed those from vehicle use under comparable annual activity assumptions providing insights into how diet and transportation choices may influence climate impact.*

### Source Data:
This project utilized two separate datasets, sourced from [Kaggle](https://www.kaggle.com/). Both native CSV files have been brought into this project.

Livestock Dataset: https://www.kaggle.com/datasets/amandaroseknudsen/gleamlivestockemissions

Vehicle Dataset: https://www.kaggle.com/datasets/brsahan/vehicle-co2-emissions-dataset

### Notebooks:
The project is supported by the execution of 5 Jupyter Notebooks that individually contributed to the final results. See below for their listing and description.
1) [emissions_analysis_livestock.ipynb](https://vscode.dev/github/dfintel25/Emissions_Analysis_Capstone/blob/main/notebooks/emissions_analysis_livestock.ipynb) | An individual Exploratory Data Analysis of the Livestock source CSV data.
2) [emissions_analysis_vehicle.ipynb](https://vscode.dev/github/dfintel25/Emissions_Analysis_Capstone/blob/main/notebooks/emissions_analysis_vehicle.ipynb) | An individual Exploratory Data Analysis of the Vehicle source CSV data.
3) [emission_comparison.ipynb](https://vscode.dev/github/dfintel25/Emissions_Analysis_Capstone/blob/main/notebooks/emission_comparison.ipynb) | A comparitive analysis of both the Livestock and Vehicle datasets.
4) [livestock_emissions_ML.ipynb](https://vscode.dev/github/dfintel25/Emissions_Analysis_Capstone/blob/main/notebooks/livestock_emissions_ML.ipynb) | An individual Machine Learning Analysis of select Livestock data features.
5) [vehicle_emissions_ML.ipynb](https://vscode.dev/github/dfintel25/Emissions_Analysis_Capstone/blob/main/notebooks/vehicle_emissions_ML.ipynb) | An individual Machine Learning Analysis of select Vehicle data features.

# Preliminary Setup Steps

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


