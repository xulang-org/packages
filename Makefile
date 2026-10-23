PYTHON ?= python3
XU_ROOT ?= $(abspath ../xu)
XU ?= $(XU_ROOT)/bin/xu
export XU_ROOT XU
.PHONY: test native versions
test:
	$(PYTHON) scripts/check_product_compiler.py
	$(PYTHON) -m unittest discover -s tests -v
native:
	$(PYTHON) sqlite/native/build.py
versions:
	$(PYTHON) scripts/module_versions.py --check
