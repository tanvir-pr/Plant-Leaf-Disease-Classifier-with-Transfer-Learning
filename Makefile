.PHONY: test app final-test

test:
	python -m pytest -q

app:
	python -m app.main

final-test:
	python scripts/run_final_test.py
