# OceanOSSE
> Python toolbox for performing Observing System Simulation Experiments (OSSEs) in ocean general circulation models.

<!-- Badges: replace URLs and slugs to match your repository and services -->
[![CI](https://github.com/AMOCcommunity/OceanOSSE/actions/workflows/ci.yml/badge.svg)](https://github.com/AMOCcommunity/OceanOSSE/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/AMOCcommunity/OceanOSSE)](LICENSE)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->
<!-- [![PyPI version](https://img.shields.io/pypi/v/<your-package>)](https://pypi.org/project/<your-package>/) -->

### **-- OceanOSSE is Under Active Development --**

## Overview

*

---

## Installation

### For Users

To get started using **OceanOSSE** create and activate a new Python virtual environment:

* Using **miniconda** or **miniforge**:

```bash
# Creating a virtual environment:
conda create -n env_oceanosse python=3.13
# Activate virtual environment <env_oceanosse>
conda activate env_oceanosse
```

* Using **venv**:

```bash
# Alternatively, creating a virtual environment using -> venv:
python -m venv env_oceanosse
# Activate virtual environment <env_oceanosse>
source /path/to/env_oceanosse/bin/activate
```

Next, install the package into the new virtual using `pip`:

```bash
# Install OceanOSSE using pip:
pip install .
```

### For Developers

Clone the repository and set up the full development environment:

```bash
git clone https://github.com/AMOCcommunity/OceanOSSE.git
cd OceanOSSE
pixi install
```

All development tasks (tests, linting, formatting, docs) are available as Pixi tasks:

```bash
# Show available tasks:
pixi task list

# Run pytest unit test suite:
pixi run tests
```

See [**CONTRIBUTING.md**](CONTRIBUTING.md) for full details on the development workflow.

---

## Usage

### Quick Start

```bash
ocean_osse run 
```

### Examples

---

## Documentation

Full documentation, including the API reference and tutorials, is available at:

**https://amoccommunity.github.io/OceanOSSE/**

---

## Contributing

Contributions are welcome! Please read [**CONTRIBUTING.md**](CONTRIBUTING.md) for details on how to set up your environment, code style expectations, and the pull request process.

To report a bug or request a feature, please [**open an issue**](https://github.com/AMOCcommunity/OceanOSSE/issues).

---

## Changelog

A full history of changes between releases is maintained in [**CHANGELOG.md**](CHANGELOG.md).

---

## Citation

If you use this software in your research, please cite it. A citation helps sustain the project and gives credit to its contributors.

**Citing the software:**

```bibtex
@software{<your-repo>,
  author    = {Last, First and Last, First},
  title     = {{<Project Name>}: <short description>},
  year      = {YYYY},
  version   = {v0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## Funding and Acknowledgements

The ongoing development of **OceanOSSE** is funded by:

- **EPOC**: [Explaining & Predicting the Ocean Conveyor](https://epoc-eu.org)


---

## License

This project is licensed under the **Apache 2.0** License — see the [LICENSE](LICENSE) file for details.
