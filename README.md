

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

running UI:
chainlit run app_ui.py -w