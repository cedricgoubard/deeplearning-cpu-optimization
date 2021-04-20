test:
	pytest --cov=src/dlcpu src/test

lint:
	pylint src/dlcpu


install:
	python3 -m venv venv
	venv/bin/pip install -U pip
	venv/bin/pip install wheel
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -e src
	venv/bin/python -m ipykernel install --user --name=dlcpu --display-name="DL CPU"