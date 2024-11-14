CONDA_ENV_NAME=vqa
PYTHON_VERSION=3.9

PYTORCH_VER="2.3.1"
TORCHVISION_VER="0.18.1"
TORCHAUDIO_VER="2.3.1"

generate_new_requirements:
	pip-compile requirements/requirements.in
	@echo "🎉 Generated new requirements.txt in requirements/ folder!"

setup_environment:
	conda create -y -c conda-forge --name ${CONDA_ENV_NAME} python=${PYTHON_VERSION} pip-tools

install_deps: install_torch
	pip install -r requirements/requirements.txt

install_torch:
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "Installing PyTorch for macOS"; \
		conda install pytorch==${PYTORCH_VER} torchvision==${TORCHVISION_VER} torchaudio==${TORCHAUDIO_VER} -c pytorch -y; \
	elif [ "$$(uname)" = "Linux" ]; then \
		if command -v nvidia-smi >/dev/null 2>&1; then \
			echo "Installing PyTorch with CUDA support for Linux"; \
			conda install pytorch==${PYTORCH_VER} torchvision==${TORCHVISION_VER} torchaudio==${TORCHAUDIO_VER} pytorch-cuda=11.8 -c pytorch -c nvidia -y; \
		else \
			echo "Installing PyTorch for Linux (CPU only)"; \
			conda install pytorch==${PYTORCH_VER} torchvision==${TORCHVISION_VER} torchaudio==${TORCHAUDIO_VER} cpuonly -c pytorch -y; \
		fi; \
	else \
		echo "Unsupported operating system"; \
		exit 1; \
	fi

reset_environment:
	conda remove --name ${CONDA_ENV_NAME} --all -y

format:
	black src