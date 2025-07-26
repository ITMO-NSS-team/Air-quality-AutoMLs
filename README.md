# Air-quality-AutoMLs

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

This work investigates automated machine learning techniques for improved air quality forecasting, specifically addressing the challenges of traditional methods requiring extensive calibration and adaptation. Focusing on PM10 concentration prediction in urban environments like Vitória, Brazil, it compares the performance of several time series models – including a moving average baseline, linear regression, FEDOT’s AutoML pipeline, and AutoNBEATS – against conventional approaches. The core methodology centers around automating model development with FEDOT while leveraging rolling forecasts for extended prediction horizons. Ultimately, this research demonstrates the potential of AutoML to achieve competitive accuracy in air quality monitoring with reduced human effort, supporting environmental policy and public health initiatives.

---

## Table of Contents

- [Content](#content)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Citation](#citation)

---
## Content

This project focuses on forecasting air quality, specifically PM10 concentration levels, using automated machine learning techniques. It leverages time series data from multiple monitoring stations to train and evaluate predictive models. The core components include data processing utilities for cleaning, preprocessing, and feature engineering of air quality measurements. Several modeling approaches are implemented: a baseline method, linear regression, an AutoML pipeline utilizing the FEDOT framework, and a neural network model (AutoNBEATS). Performance is assessed via standard error metrics, with visualizations aiding in result interpretation. The project aims to demonstrate the efficacy of automated methods for scalable and adaptive air quality monitoring, reducing reliance on manual model development.

---

## Algorithms

This project employs several computational techniques for air quality forecasting. A foundational approach involves time series analysis using moving averages and linear regression to establish baseline predictions. More advanced methods leverage automated machine learning (AutoML) with the FEDOT framework, which automatically designs and optimizes predictive models. Neural network-based approaches, specifically AutoNBEATS, are also utilized for forecasting. These algorithms aim to predict pollutant concentrations by identifying patterns in historical data. The core objective is to automate model development, reduce human effort, and achieve competitive accuracy compared to traditional methods, ultimately supporting environmental monitoring and public health initiatives.

---

## Installation

Install Air-quality-AutoMLs using one of the following methods:

**Build from source:**

1. Clone the Air-quality-AutoMLs repository:
```sh
git clone https://github.com/ITMO-NSS-team/Air-quality-AutoMLs
```

2. Navigate to the project directory:
```sh
cd Air-quality-AutoMLs
```

---

## Citation

If you use this software, please cite it as below.

### APA format:

    ITMO-NSS-team (2025). Air-quality-AutoMLs repository [Computer software]. https://github.com/ITMO-NSS-team/Air-quality-AutoMLs

### BibTeX format:

    @misc{Air-quality-AutoMLs,

        author = {ITMO-NSS-team},

        title = {Air-quality-AutoMLs repository},

        year = {2025},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/ITMO-NSS-team/Air-quality-AutoMLs.git}},

        url = {https://github.com/ITMO-NSS-team/Air-quality-AutoMLs.git}

    }

---
