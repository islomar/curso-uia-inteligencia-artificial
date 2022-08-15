.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help.
	@grep -E '^\S+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-30s %s\n", $$1, $$2}'

build-jupyter-image: ## Build Jupyter image
	docker build -t jupyter jupyter-dockerfile

run-jupyter: ## Run Jupyter
	docker compose -f jupyter/docker-compose.yml up
	#docker run -p 8888:8888 -v ${PWD}/ipynbs:/workspace/ipynbs jupyter jupyter notebook --allow-root --no-browser --ip=0.0.0.0

run-pytorch: ## Run pytorch
	docker compose -f pytorch/docker-compose.yml up