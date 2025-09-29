.DEFAULT_GOAL := all

venv:  
	python -m venv .venv
	@echo "Created venv at .venv"

install: ## Install project dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed"

freeze: ## Save current environment to requirements.txt
	pip freeze > requirements.txt
	@echo "requirements.txt updated"

prepare: ## Build datasets (load -> pose_csv -> angles_csv)
	python -m src.runner prepare

train: ## Train the model (runs prepare if needed)
	python -m src.runner train

test: ## Run webcam test (press 'q' to quit)
	python -m src.runner test

detect: ## Detect pose in a single image: make detect IMG=path/to/img.jpg
ifndef IMG
	$(error Usage: make detect IMG=path/to/image.jpg)
endif
	python -m src.runner detect --img "$(IMG)"

clean: ## Remove generated files and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache data models
	rm -f src/pose_dataset.csv src/pose_angles_dataset.csv
	@echo "Cleaned artifacts"

all: prepare train test
