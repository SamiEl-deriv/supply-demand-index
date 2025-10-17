<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/regentmarkets/quants">
    <img src="img/deriv_red_logo.png" alt="Logo" width="240" height="150">
  </a>

<h3 align="center">Deriv Quant Package</h3>

  <p align="center">
    A comprehensive quantitative finance toolkit for Deriv's trading systems
    <br />
    <a href="https://github.com/regentmarkets/quants"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/regentmarkets/quants/issues">Report Bug</a>
    ·
    <a href="https://github.com/regentmarkets/quants/issues">Request Feature</a>
  </p>
</div>

    

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to gather and organize all the different pieces of code created by the quantitative finance team at Deriv. The repository is structured as a comprehensive Python package that provides tools for options pricing, index building, risk analysis, and more.

The repository is organized into several main components:

1. **utility**: The core Python package containing:
   - **indexbuilder**: Tools for building and managing synthetic and financial indices
   - **model_validation**: Tools for validating models and testing implementations
   - **pricer**: Options pricing engines including BSM, PDE, and Monte Carlo methods
   - **RiskAnalysis**: Tools for analyzing financial time series data and risk metrics
   - **useful_scripts**: Utility scripts for data retrieval, account management, and more

2. **documentation**: Detailed specifications for various products and implementations

3. **RandD**: Research and development projects including:
   - Backtesting frameworks
   - Index feed simulations
   - Specialized product pricing (Snowballs, Accumulators, etc.)

### Repository Structure

Below is a high-level overview of the repository structure:

```
.
├── RandD
│   ├── Backtesting
│   ├── Contest
│   ├── DEX
│   ├── Index_feed
│   ├── Snowball
│   ├── VSI
│   ├── copulas_products
│   ├── live_Trading
│   ├── stable_spread
│   └── weekend_indices
├── utility
│   ├── RiskAnalysis
│   ├── indexbuilder
│   ├── model_validation
│   ├── pricer
│   ├── tests
│   └── useful_scripts
├── documentation
│   ├── POC
│   ├── backend_product_specification
│   ├── exploits
│   └── wikijs
└── img


# Documentation Details

The documentation is organized into several key areas:

## wikijs
Contains comprehensive documentation on:
- **derived_underlying**: Documentation for various index types
  - Basket Indices
  - Derived FX
  - Synthetic Indices (Bull/Bear, Crash/Boom, DEX, Drift Switch, Jump, Range Break, Step, Volatility)
- **products**: Documentation for various products
  - Vanilla and digital options (American, Asian, European)
  - Barrier options and specialized products (Accumulator, Turbo, etc.)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
* [Python][python-url]
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

Here are the key dependencies:

#### Core Dependencies
* [numpy][numpy-url] - Used throughout for numerical computations
* [pandas][pandas-url] - Used for data manipulation and analysis
* [scipy][scipy-url] - Used for statistical functions and distributions
* [opencv-python][opencv-python-url] - Used for image processing

#### Module-Specific Dependencies
* [plotly][plotly-url] - Used in IndexBuilder for visualization
* [psycopg2][psycopg2-url] - Used for database connections in feed retrieval scripts
* [matplotlib][matplotlib-url] - Used for visualization in RiskAnalysis

#### Optional Dependencies
* [tqdm][tqdm-url] - For progress bars in long-running operations

Any of these dependencies can be installed using pip with the following syntax in your local terminal:

```sh
pip install package_name
```

### Installation

The package can be installed in several ways:

#### Method 1: Install from the repository

1. Clone the repository:
   ```sh
   git clone https://github.com/regentmarkets/quants.git
   ```

2. Navigate to the repository directory:
   ```sh
   cd quants
   ```

3. Install the package in development mode:
   ```sh
   pip install -e .
   ```
   This will install the package as `deriv_quant_package` while allowing you to modify the source code.

#### Method 2: Update an existing installation

If you have already cloned the repository, make sure you have the latest version:
   ```sh
   git pull
   ```

Then reinstall the package:
   ```sh
   pip install -e .
   ```

#### Method 3: Install specific components

You can also install only specific components of the package:

```sh
# Install with optional dependencies
pip install -e ".[optional]"

# Install with development dependencies
pip install -e ".[dev]"

# Install with both optional and development dependencies
pip install -e ".[optional,dev]"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Once the package is installed, you can import it in any Python script. Each module can be imported separately:

```python
# Import specific modules
from deriv_quant_package.pricer import *
from deriv_quant_package.indexbuilder import IndexBuilder
from deriv_quant_package.RiskAnalysis import IndexAnalysis
from deriv_quant_package.model_validation import ModelValidator

# Or import the entire package
import deriv_quant_package
```

### Main Modules

#### 1. Pricer
Options pricing engines with support for:
- Black-Scholes-Merton model
- PDE (finite difference) methods
- Monte Carlo simulations
- Various option types (vanilla, digital, barrier, etc.)

#### 2. IndexBuilder
Tools for building and managing indices:
- Synthetic index generation
- FX derived indices
- Financial data feeds
- Statistical analysis

#### 3. RiskAnalysis
Comprehensive tools for analyzing financial time series:
- Return distribution analysis
- Volatility calculations
- Drawdown visualization
- Moment analysis (skewness, kurtosis)

#### 4. Model Validation
Tools for validating and testing models:
- Benchmark testing
- Comparison with analytical solutions
- Error analysis and visualization
- Performance evaluation

#### 5. Useful Scripts
Utility scripts for various operations:
- Data retrieval from QA environments
- Account management
- Service monitoring
- Volatility surface tools

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

### Current Status
- [x] Core package structure and installation
- [x] Basic documentation and examples

### In Development
- **Pricers**
  - [x] Black-Scholes-Merton implementation
  - [x] PDE with flat volatility
  - [ ] PDE with local volatility (in progress)
  - [ ] PDE with stochastic volatility
  - [ ] Monte Carlo engine enhancements

- **Index Builder**
  - [x] FX derived engine
  - [x] Financial & synthetic data fetcher
  - [ ] Synthetic data generation framework
  - [ ] Integration with Monte Carlo module

- **Volatility Surfaces**
  - [x] Flat volatility model
  - [x] Spline interpolation
  - [x] Vanna-Volga approach
  - [ ] Advanced calibration methods

- **Research & Development**
  - [ ] Enhanced backtesting framework
  - [ ] Multi-asset correlation models

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion, please feel free to suggest it.  
Suggestions can either be modifications of existing codes or addition of new modules, classes or functions.  
In any case, before making any contribution to the project:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Do the modifications/additions you want to do 
4. Add your work to your staging area (`git add .`)
5. Commit your work: `git commit -m 'Add some AmazingFeature'` (do not forget to add a comment)
6. Push your branch to your remote (`git push your_remote feature/AmazingFeature`)
7. Open a Pull Request 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Internal Deriv python package. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Victor Dagard - victor.dagard@deriv.com

Project Link: [https://github.com/regentmarkets/quants](https://github.com/regentmarkets/quants)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* The Deriv Quantitative Finance Team
* Contributors to the various modules and tools
* Open source libraries that make this package possible

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[forks-url]: https://github.com/regentmarkets/quants/network/members


[python-img]: https://www.python.org/static/community_logos/python-logo-master-v3-TM.png 
[python-url]: https://www.python.org/
[opencv-python-url]: https://pypi.org/project/opencv-python/
[numpy-url]: https://numpy.org/
[pandas-url]: https://pandas.pydata.org/
[psycopg2-url]: https://www.psycopg.org/docs/
[scipy-url]: https://scipy.org/
[plotly-url]: https://plotly.com/python/
[matplotlib-url]: https://matplotlib.org/
[setuptools-url]: https://setuptools.pypa.io/en/latest/
[tqdm-url]: https://tqdm.github.io/
[wheel-url]: https://pythonwheels.com/

[contributors-shield]: https://img.shields.io/github/contributors/regentmarkets/quants.svg?style=for-the-badge
[contributors-url]: https://github.com/regentmarkets/quants/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/regentmarkets/quants.svg?style=for-the-badge
[stars-shield]: https://img.shields.io/github/stars/regentmarkets/quants.svg?style=for-the-badge
[stars-url]: https://github.com/regentmarkets/quants/stargazers
[issues-shield]: https://img.shields.io/github/issues/regentmarkets/quants.svg?style=for-the-badge
[issues-url]: https://github.com/regentmarkets/quants/issues
[license-shield]: https://img.shields.io/github/license/regentmarkets/quants.svg?style=for-the-badge
[license-url]: https://github.com/regentmarkets/quants/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
