# 10 Academy: Artificial Intelligence Mastery


# Telecom Analytics Project

## Project Overview

This project is focused on performing a comprehensive analysis of user behavior, engagement, experience, and satisfaction in a telecom dataset. The project is designed with modular, reusable code and features a Streamlit dashboard for data visualization. The key objectives include:

- **User Overview Analysis**: Analyze handset usage, handset manufacturers, and application usage.
- **User Engagement Analysis**: Track user engagement across different applications and cluster users based on engagement metrics.
- **Experience Analytics**: Assess user experience based on network parameters and device characteristics.
- **Satisfaction Analysis**: Calculate and predict user satisfaction scores based on engagement and experience.

The project structure is organized to support reproducible and scalable data processing, modeling, and visualization.

## Project Structure

```plaintext
├── .vscode/
│   └── settings.json                # Configuration for VSCode environment
├── .github/
│   └── workflows/
│       ├── unittests.yml            # GitHub Actions workflow for running unit tests
├── .gitignore                        # Files and directories to be ignored by Git
├── requirements.txt                  # List of dependencies for the project
├── README.md                         # Project overview and instructions
├── Dockerfile                        # Instructions to build a Docker image
├── scripts/
│   ├── __init__.py
│   ├── data_preparation.py           # Script for data cleaning and preparation
│   ├── eda_pipeline.py               # EDA steps implemented using scikit-learn pipeline
│   ├── feature_store.py              # Code for interacting with SQL database
│   ├── models.py                     # Machine learning models and training scripts
│   ├── dashboard.py                  # Streamlit dashboard script
│   ├── utils.py                      # Utility functions
├── notebooks/
│   ├── __init__.py
│   ├── exploratory_analysis.ipynb    # Jupyter notebook for initial EDA and analysis
│   ├── model_training.ipynb          # Jupyter notebook for model training and evaluation
│   ├── README.md                     # Description of notebooks
├── tests/
│   ├── __init__.py
│   ├── test_data_preparation.py      # Unit tests for data preparation module
│   ├── test_eda_pipeline.py          # Unit tests for EDA pipeline module
│   ├── test_feature_store.py         # Unit tests for SQL feature store interactions
│   ├── test_models.py                # Unit tests for models and training scripts
│   ├── test_dashboard.py             # Unit tests for Streamlit dashboard
└── src/
    ├── __init__.py
    ├── data_preparation_script.py    # Script for running data preparation independently
    ├── eda_pipeline_script.py        # Script for running EDA pipeline independently
    ├── model_training_script.py      # Script for running model training independently
    ├── dashboard_deploy.py           # Script for deploying Streamlit dashboard
    └── README.md                     # Description of scripts
