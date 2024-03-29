VENV=.venv
PYTHON=$(VENV)/bin/python3
DOCKER=minivae
PIP_FREEZE=.requirements.freeze.txt
TEST_DIR=minivae/
PY_FILES=minivae/

.PHONY: ci
ci: $(PY_FILES) py-deps type-check lint test

.PHONY: type-check
type-check: $(VENV) $(PY_FILES)
	$(PYTHON) -m mypy \
		--install-types \
		--non-interactive \
		--ignore-missing-imports \
		--pretty \
		$(PY_FILES)

.PHONY: lint
lint: $(VENV) $(PY_FILES)
	$(PYTHON) -m isort $(PY_FILES)
	$(PYTHON) -m flake8 $(PY_FILES) \
		--ignore E402,W503,E731


.PHONY: test
test: $(VENV) $(PY_FILES)
	$(PYTHON) -m pytest $(TEST_DIR)

.PHONY: py-deps
py-deps: $(PIP_FREEZE)

$(PIP_FREEZE): $(VENV) requirements.txt
	$(PYTHON) -m pip install \
		--upgrade \
		--require-virtualenv \
		pip
	$(PYTHON) -m pip install \
		--upgrade \
		--require-virtualenv \
		-r requirements.txt
	$(PYTHON) -m pip freeze > $(PIP_FREEZE)

$(VENV):
	python3 -m venv $(VENV)

.PHONY: docker
docker: docker-build
	docker run \
		--rm \
		-it \
		--gpus all \
		-v $(PWD):/workdir/ \
		$(DOCKER) bash

docker-build: Dockerfile requirements.txt
	docker build -t $(DOCKER) .

.PHONY: clean
clean:
	rm -rf \
		$(VENV) \
		$(PIP_FREEZE) \
		.mypy_cache/
