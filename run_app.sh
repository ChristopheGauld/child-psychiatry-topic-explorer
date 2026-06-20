#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/.packages${PYTHONPATH:+:$PYTHONPATH}"
exec streamlit run app.py
