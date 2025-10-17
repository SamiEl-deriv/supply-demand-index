.. API Automation Testing Project documentation master file, created by
   sphinx-quickstart on Thu Aug  1 13:33:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Model Validation Teams's API Automation Testing Project's documentation!
===================================================================================
Overview
========
The **API Testing Automation Library** enables automated validation of trading products using the Deriv API. It supports various products, including Vanilla, Rise/Fall, Multiplier, Turbo, and Accumulator contracts. The library features **GUI integration** for streamlined testing, dynamic data export capabilities, and command-line execution for flexible usage.

It simplifies:

- Switching product parameters.

- Validating pricing mechanisms with prebuilt financial models.

- Exporting data to Excel for recording. 

The user-friendly design and a smooth setup process via `requirements.txt` ensure compatibility across environments.

Key Features
============
- **Support for Multiple Trading Products**:  

  Validate Rise/Fall, Vanilla, Turbo, Multiplier, and Accumulator contracts.

- **Dynamic Validation**: 

  Automates validation for payouts, barriers, bids, and ask prices using robust financial models.

- **Graphical User Interface (GUI)**:  

  A customizable GUI allows you to set parameters interactively, test features, and analyze results.

- **Automated Analysis**:  

  Generates structured statistics in formats like `pandas` DataFrame or Excel files.

- **Data Export Tools**:  

  Logs contract details, WebSocket messages, and validation data into Excel spreadsheets.

Getting Started
===============

Instructions to Clone the Repository
-------------------------------------

1. **Open Terminal**:
   - On macOS, press `Cmd + Space`, type "Terminal" in the search bar, and hit Enter.

2. **Navigate to the Directory** where you'd like to clone the repository (optional):
   - Use the `cd` command to go to the folder where you want the repository to be cloned:

     .. code-block:: bash

        cd /path/to/your/desired/folder/

3. **Clone the Repository**:
   - Replace `<repository-url>` with the actual Git repository URL (e.g., GitHub URL) and run the following command:

     .. code-block:: bash

        git clone <repository-url>

   Example:

     .. code-block:: bash

        git clone https://github.com/regentmarkets/quants-model-validation.git

4. **Navigate into the Cloned Repository**:

     .. code-block:: bash

        cd quants-model-validation

The repository is now cloned to your local machine, and you can proceed with further setup steps such as installing dependencies or running the scripts.

Instructions to Install Dependencies from `requirements.txt`
-------------------------------------------------------------

1. **Open Terminal**:
   - On macOS, press `Cmd + Space`, type "Terminal" in the search bar, and hit Enter.

2. **Navigate to Your Project Directory**:
   - Use the `cd` command to move to the directory where your project files (including `requirements.txt`) are located. Replace `/path/to/your/project/` with the actual folder path on your machine.

     .. code-block:: bash

        cd /path/to/your/project/

   Example:

     .. code-block:: bash

        cd ~/Documents/Repos/Personal/1\ API\ framework/

3. **Install the Required Packages**:
   - Run the following command to install all the dependencies listed in `requirements.txt`:

     .. code-block:: bash

        pip install -r requirements.txt

Instructions to Use the Example Script in a Jupyter Notebook
------------------------------------------------------------

1. **Open the Example Script**:

   - Locate and open the file named `Example Script.ipynb` in your Jupyter Notebook environment.
   - The file should resemble the layout below when opened:

      .. image:: _static/image1.png
         :alt: Jupyter Notebook Example Script

2. **Getting Started**:

   - The script contains examples for:
     - Importing the necessary libraries.
      .. code-block:: python

         import mvapi as mv

     - Initializing an instance of a product class.
     - Running basic functions, such as generating a price proposal or validating contract outputs.
   - Refer to the `Getting Started` section of the notebook for structured, step-by-step instructions.

3. **Creating a Product Instance**:

   - Follow this syntax to create a product instance as demonstrated:

     .. code-block:: python

        product = mv.Turbo()  # The instance of a Turbo class

   - By default:

     - **Endpoint**: The production endpoint `wss://blue.derivws.com/websockets/v3?app_id=16929` is used.

     - **Contract Parameters**:
       - There are preset default values for parameters such as stake or contract_type unless specified.
     - **API Token**: There is no pre-configured API token for security reasons; one must be provided through the GUI or coded configuration.

4. **Using the Notebook for Validation**:

   - Execute the provided cells sequentially:
     - Import the library as `mv`.
     - Initialize the product instance.
     - Use functions like `price_proposal()` or `check_payout()` to validate contracts.

   Example:

     .. code-block:: python

        product.check_payout()

5. **Script Details**:

   - The **Example Script** provides comprehensive guidance on:
     - Running checks for payouts, bids, and ask prices for different product types.
     - Performing contract randomization.
     - Exporting validation results to Excel.

   - Feel free to modify and extend the script for your specific testing needs.

Instructions to Run the Script by Double-Clicking (via `.command` File)
-----------------------------------------------------------------------

1. **Make the `.command` File Executable**:
   - Open **Terminal** and navigate to the folder where the `.command` file is located:

     .. code-block:: bash

        cd /path/to/your/project/

   - Make the `.command` file executable by running the following command:

     .. code-block:: bash

        chmod +x run_mvapi.command

2. **Run the `.command` File**:
   - Open **Finder** and navigate to the project folder where both the `run_mvapi.command` and `mvapi.py` files are located.
   - **Double-click** the `run_mvapi.command` file.

   This will launch the script in a new **Terminal** window, and `mvapi.py` will run automatically.


Instructions to Run the Script via Terminal (Command Line)
----------------------------------------------------------

1. **Open Terminal**:
   - On macOS, press `Cmd + Space`, type "Terminal" in the search bar, and hit Enter.

2. **Navigate to Your Project Directory**:
   - Use the `cd` command to move to the directory where your project files are located (replace `/path/to/your/project/` with the actual folder path on your machine).

     .. code-block:: bash

        cd /path/to/your/project/

   Example:

     .. code-block:: bash

        cd ~/Documents/Repos/quants-model-validation/Other Projects/API Testing Automation Project/

3. **Run the Python Script** using Python 3:

     .. code-block:: bash

        python3 mvapi.py



.. Documentation Contents
.. ======================
.. [work in progress]

.. .. toctree::
..    :maxdepth: 2
..    :caption: Contents:

Module Documentation
====================

.. automodule:: mvapi
   :members:
   :undoc-members:
   :show-inheritance:

.. .. autoclass:: mvapi.Product
..    :members: __init__, _set_attributes, _set_app_details, _get_app_markup, _get_app_id, _proposal, _send, _check_connection, _check_closed, _recv_msg_handler
..    :undoc-members: 
..    :show-inheritance:


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
