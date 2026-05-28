"""
clean_data.py

Data-cleaning module for the meningioma-atypier project.

This file converts the raw meningioma dataset into a clean, analysis-ready
table for exploratory analysis, statistical modeling, and the final Streamlit
calculator.

Main responsibilities:
- load the raw dataset
- standardize column names
- normalize categorical values
- handle missing or unclear entries
- define the main outcome label: atypical vs non-atypical meningioma
- validate expected columns and data types
- export the cleaned dataset

Important:
No modeling happens here. This file only prepares trustworthy data.
"""

