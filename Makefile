# xorgate-model-forge developer targets. `uv` manages the environment
# (`uv sync --extra all --extra dev` creates .venv from uv.lock); `npm ci`
# installs the engine for the golden cases that run the real `xwe`.
UV ?= uv
XMF_IMAGE ?= ghcr.io/epyelabs/xmf
VERSION := $(shell $(UV) run python -c 'import xorgate_forge; print(xorgate_forge.__version__)')

.PHONY: help setup lint typecheck test schemas fixtures fixture-dataset images image-cpu image-cpu-multiarch image-cuda clean

help:
	@echo "setup            uv sync (all extras + dev) and npm ci"
	@echo "lint             ruff check + ruff format --check"
	@echo "typecheck        mypy (strict) over src/"
	@echo "test             the golden suite (pulls the fixtures first)"
	@echo "schemas          re-emit contracts/*.schema.json from the pydantic models and the engine"
	@echo "fixtures         pull the golden video and model per fixtures/fixtures.json"
	@echo "fixture-dataset  rebuild fixtures/ds from drive.mp4 with the pulled yolov8n"
	@echo "images           build the cpu image for the native architecture (see image-cpu-multiarch, image-cuda)"

setup:
	$(UV) sync --python 3.11 --extra all --extra dev
	npm ci --no-audit --no-fund

lint:
	$(UV) run ruff check src scripts
	$(UV) run ruff format --check src scripts

typecheck:
	$(UV) run mypy

fixtures:
	$(UV) run python scripts/fetch_fixtures.py

test: fixtures
	$(UV) run python scripts/golden_test.py

schemas:
	$(UV) run python scripts/emit_schemas.py
	node scripts/vendor-manifest-schema.mjs

fixture-dataset: fixtures
	$(UV) run python scripts/build_fixture_dataset.py

images: image-cpu

image-cpu:
	docker build --target cpu -t $(XMF_IMAGE):dev -t $(XMF_IMAGE):$(VERSION) .

image-cpu-multiarch:
	docker buildx build --target cpu --platform linux/amd64,linux/arm64 -t $(XMF_IMAGE):$(VERSION) .

image-cuda:
	docker buildx build --target cuda --platform linux/amd64 -t $(XMF_IMAGE):$(VERSION)-cuda --load .

clean:
	rm -rf out out-* .mypy_cache .ruff_cache
