TEST_PATH=./

install:
	python setup.py install
build:
	python setup.py sdist bdist_wheel
clean:
	make clean-pyc
	make clean-build
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

test: clean-pyc
	python -m unittest discover -s $(TEST_PATH)

run:
	python bin/main.py

help:
	@echo "    clean-pyc"
	@echo "        Remove python artifacts."
	@echo "    clean-build"
	@echo "        Remove build artifacts."
	@echo "    isort"
	@echo "        TODO: Sort import statements."
	@echo "    lint"
	@echo "        TODO: Check style with flake8."
	@echo "    test"
	@echo "        Run py.test"
	@echo '    run'
	@echo '        Run the `LAADS` app on your local machine.'
	@echo '    docker-run'
	@echo '        TODO: Build and run the `LAADS` service in a Docker container.'
