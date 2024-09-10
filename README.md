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
│   ├── data_processing.py           # Script for data cleaning and processing
│   ├── db_connect.py                # Script for database connection engine
│   ├── eda_pipeline.py              # EDA steps implemented 
│   ├── experience_analytics.py      # User experience analytics module
│   ├── handset_analysis.py          # Handset analysis module
│   ├── habdset_dashboard.py         # Streamlit dashboard for visualizing top handsets
│   ├── satisfaction_analytics.py     # Machine learning models and training scripts for predicting satisfaction score
│   ├── satisfaction_dashabord.py     # Streamlit dashboard script
│   ├── user_analysis.py               # User analysis functions
│   ├── user_engagement_analysis.py     # User engagement analytics module
│   ├── user_engagement_dashboard.py     # User engagement dashboard module
├── notebooks/
│   ├── __init__.py
│   ├── experience_analysis.ipynb    # Jupyter notebook for user experience analysis
│   ├── user_analaysis_notebook.ipynb          # Jupyter notebook for user and handset analysis
│   ├── user_engagement_notebook.ipynb          # Jupyter notebook for user engagement analysis
│   ├── usersatisfaction_analytics_notebook.ipynb          # Jupyter notebook for user satisfaction score prediction
│   ├── README.md                     # Description of notebooks
├── tests/
│   ├── __init__.py
│   ├── test_user_engagement_analysis.py      # Unit tests for user engagement module
│   ├── test_user_analysis.py         # Unit tests for user analysis module
│   ├── test_handset_analysis.py      # Unit tests for handset analysis module
│   ├── test_experience_analytics.py  # Unit tests for user experience module
│   ├── test_eda_pipeline.py          # Unit tests for EDA pipeline module
│   
└── src/
    ├── __init__.py
    ├── app.py    # Script for running streamlit dashboard
    └── README.md                     # Description of scripts
