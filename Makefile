.PHONY: fixture install install-models test lint clean

fixture:
	python scripts/generate_sample_fixture.py

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

install-models:
	python -m pip install -r requirements-models.txt
	python -m pip install -e .

test:
	python -m pytest -m "not network" -q

lint:
	python -m ruff check src tests scripts

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p) for p in (Path('.pytest_cache'), Path('.ruff_cache')) if p.exists()]"

