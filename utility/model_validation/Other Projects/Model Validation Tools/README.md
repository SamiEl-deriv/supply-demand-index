<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/regentmarkets/quants-model-validation">
    <img src="img/deriv_red_logo.png" alt="Logo" width="240" height="150">
  </a>

<h3 align="center">mvlib quant package</h3>

  <p align="center">
    A Model Validation Tools Python Package
    <br />
    <a href="https://github.com/regentmarkets/quants-model-validation"><strong>Explore the docs (COMING SOON) »</strong></a>
    <br />
    <br />
   <!-- <a href="https://github.com/regentmarkets/quants-model-validation">View Demo</a>
    ·-->
    <a href="https://github.com/regentmarkets/quants-model-validation/issues">Report Bug</a>
    ·
    <a href="https://github.com/regentmarkets/quants-model-validation/issues">Request Feature</a>
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

This project was made to unify our Model Validation code used in all our reports. We also have plans to allow live Production testing of the API.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
* [![Python][python-img]][python-url]
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

The required packages can be found within `requirements.txt`.

To install them in one go, use the shell command below:
```sh
pip install -r requirements.txt
```

### Installation

* Clone the repo: 
   ```sh
   git clone https://github.com/regentmarkets/quants-model-validation.git
   ```
* Or Git Pull (Fetch + Merge if you prefer):
   ```sh
   git pull
   ```
* Once you have the latest version, navigate to this folder in your local repo and use the following shell command to install the package:  
   ```sh
   pip install ./
   ```

If you would like to make an editable install, use the following command:
```sh
pip install -e ./
```

In the editable mode, you can edit and run your code without having to reinstall the package. There are some [caveats](https://setuptools.pypa.io/en/latest/userguide/development_mode.html#limitations) (Especially with [Jupyter Notebooks](https://stackoverflow.com/questions/47902297/avoid-restarting-jupyter-kernel-in-package-develop-mode)).

If VSCode does not properly autocomplete in an editable install, open `settings.json` (`CTRL+SHIFT+P` then type "Open User Settings (JSON)") and add the entry:

```json
"python.analysis.extraPaths": [
        ".../quants-model-validation/Other Projects/Model Validation Tools/mvlib-indices/src",
        ".../quants-model-validation/Other Projects/Model Validation Tools/mvlib-api/src"
    ]
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Once the package is installed, you can import it any PY script or Jupyter notebook:

```py
import mvlib
import mvlib.api as api
from mvlib.indices.tactical.index import Tactical
```

<!--  # (For more examples, please refer to the [Documentation](https://example.com)_ )  -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- See the [open issues](https://github.com/regentmarkets/quants/issues) for a full list of proposed features (and known issues). -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
<!-- ## Contributing

If you have a suggestion, please feel free to suggest it.  
Suggestions can either be modifications of existing codes or addition of new modules, classes or functions.  
In any case, before making any contribution to the project

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Do the modifications/additation  you want to do 
4. Add your work to your staging area `git add .`
4. Commit your work : `git commit -m 'Add some AmazingFeature'` (do not forget to add a comment)
6. Push your branch to your remote (`git push your_remote feature/AmazingFeature`)
6. Open a Pull Request 

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- LICENSE -->
<!-- ## License

Internal Deriv python package. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT -->
## Contact

Vishal Menon - vishal.menon@deriv.com
Matthew Chan - matthew.chan@deriv.com

Project Link: [GitHub](https://github.com/regentmarkets/quants-model-validation/tree/master/Other%20Projects/Model%20Validation%20Tools)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS
## Acknowledgments

* []()
* []()
* []() -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[forks-url]: https://github.com/regentmarkets/quants-model-validation/network/members


[python-img]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff
[python-url]: https://www.python.org/