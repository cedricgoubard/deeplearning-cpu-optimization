test:
	pytest --cov=src/dlcpu src/test

lint:
	pylint src/dlcpu
