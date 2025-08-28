#!/bin/bash
# -----------------------------
# LLM Financial Analyst - Quick Start
# -----------------------------

# Check if venv exists
VENV_PATH="envs/finance-venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH"
    echo "Please create it first with: python3 -m venv envs/finance-venv"
    exit 1
fi

# Activate the virtual environment
source $VENV_PATH/bin/activate

# Reminder message
echo "------------------------------------------"
echo "LLM Financial Analyst"
echo "Virtual environment activated: $VENV_PATH"
echo "Launching Streamlit app..."
echo "------------------------------------------"

# Launch Streamlit
streamlit run app.py
