# Design Notes — California Housing Value Lab

## App design concept

The app uses a **California coastal field-guide** style: warm sand backgrounds, deep Pacific blue navigation, orange/gold accent colors, large editorial typography, and rounded analytical cards. The goal is to make the project feel like a polished business-support tool rather than a plain class notebook.

## Rubric-centered design choices

- Sidebar navigation exactly supports the required Streamlit structure.
- The first page explains the business case and data overview.
- The visualization page uses interactive Plotly charts and a Seaborn/Matplotlib backup visual.
- The model page uses scikit-learn Linear Regression with metrics, diagnostics, coefficients, and a simulator.
- The final guide page supports the 8-minute presentation requirement.

## Presentation advantage

The interface is intentionally built around a story flow: **business problem → dataset → visual evidence → model performance → simulator → limitations**. That makes it easier to present confidently and avoid sounding like you are only describing code.
