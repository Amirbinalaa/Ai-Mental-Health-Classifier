#!/bin/bash
# Start Flask backend in the background
gunicorn --bind 0.0.0.0:5000 app:app &

# Start Streamlit frontend on port 7860 (Hugging Face default)
streamlit run src/frontend/app.py --server.port 7860 --server.address 0.0.0.0
