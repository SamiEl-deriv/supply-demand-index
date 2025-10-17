# API Testing Automation Library 
## Project Description 
The API Testing Automation Library automates the testing and validation of products using the Deriv API. Initially focused on vanilla products, the library now supports Vanilla, Rise/Fall, Multiplier, Turbo and Accumulator, with added features such as GUI integration for easy testing, data export, and command-line execution. The tool simplifies switching product parameters, validating pricing, and managing accounts through a user-friendly interface. Dependencies can be easily installed via requirements.txt for a streamlined setup on different environments. 

---

### Instructions to Clone the Repository

1. **Open Terminal**:
   - On macOS, press `Cmd + Space`, type "Terminal" in the search bar, and hit Enter.

2. **Navigate to the Directory** where you'd like to clone the repository (optional):
   - Use the `cd` command to go to the folder where you want the repository to be cloned:

   ```bash
   cd /path/to/your/desired/folder/
   ```

3. **Clone the Repository**:
   - Replace `<repository-url>` with the actual Git repository URL (e.g., GitHub URL) and run the following command:

   ```bash
   git clone <repository-url>
   ```

   Example:
   
   ```bash
   git clone https://github.com/regentmarkets/quants-model-validation.git
   ```

4. **Navigate into the Cloned Repository**:

   ```bash
   cd quants-model-validation
   ```

The repository is now cloned to your local machine, and you can proceed with further setup steps such as installing dependencies or running the scripts.

---


### Instructions to Install Dependencies from `requirements.txt`

1. **Open Terminal**:
   - On macOS, press `Cmd + Space`, type "Terminal" in the search bar, and hit Enter.

2. **Navigate to Your Project Directory**:
   Use the `cd` command to move to the directory where your project files (including `requirements.txt`) are located. Replace `/path/to/your/project/` with the actual folder path on your machine.

   ```bash
   cd /path/to/your/project/
   ```

   Example:

   ```bash
   cd ~/Documents/Repos/Personal/1\ API\ framework/
   ```

3. **Install the Required Packages**:
   Run the following command to install all the dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

---
### Instructions to Run the Script by Double-Clicking (via `.command` File)

1. **Make the `.command` File Executable**:
   - Open **Terminal** and navigate to the folder where the `.command` file is located:
   
     ```bash
     cd /path/to/your/project/
     ```

   - Make the `.command` file executable by running the following command:

     ```bash
     chmod +x run_mvapi.command
     ```

2. **Run the `.command` File**:
   - Open **Finder** and navigate to the project folder where both the `run_mvapi.command` and `mvapi.py` files are located.
   - **Double-click** the `run_mvapi.command` file.

   This will launch the script in a new **Terminal** window, and `mvapi.py` will run automatically.

---


### Instructions to Run the Script via Terminal (Command Line)

1. **Open Terminal**:
   - On macOS, press `Cmd + Space`, type "Terminal" in the search bar, and hit Enter.

2. **Navigate to Your Project Directory**:
   Use the `cd` command to move to the directory where your project files are located (replace `/path/to/your/project/` with the actual folder path on your machine).

   ```bash
   cd /path/to/your/project/
   ```

   Example:
   
   ```bash
   cd ~/Documents/Repos/quants-model-validation/Other Projects/API Testing Automation Project/
   ```
   
3. **Run the Python Script** using Python 3:

   ```bash
   python3 mvapi.py
   ```

---


### Project Update Log:
Milestones achieved as of 04/06/24:
* Create the fundamental script with the capability to:
  * Request and display price proposals
  * Buy contracts
  * Store contracts/proposals obtained to be validated
  * Change current product type/product parameters on the fly
  * Change API endpoints/API tokens used on the fly
* Vanilla option validation:
  * Barrier validation (done)
  * Ask price check (done)
  * Bid price check (in-progress)
  * Close at expiry check (in-progress)

Milestones achieved as of 19/06/24:
* Improved error handling
  * Automatically handles connection issues with websocket
  * Automatically handles more contract validation errors (e.g. invalid stake/payout per point)
* Turbo option validation:
  * payout per points choices validation

Milestones achieved as of 01/07/24:
* Add Turbo validation functions
  * Add validation of pricing for sold, expired, knocked out and currently open contracts
  * improve payout per point choices and barrier validation functions
* Overall project
  * Refractor validation functions and error handling functions
  * Improved overall error handling

Milestones achieved as of 02/07/24:
  * Overall Project
    * Project has been exported to a .py python module (mvapi.py) and can now be imported
    * Refractored and cleaned up code
  * Accumulators
    * Tick size barrier for accumulators are no longer hard coded and are calculated based on given loss probabilities
    * Ability to change the loss probabilities and recalculate tick size barrier accordingly
   
Milestones achieved as of 29/07/24:
* Overall Project
   * Adds support to detect app markup on 3rd party apps.
   * Improved validation results display.
   * Refactored constants for better code organisation.
* Product Class
    * Refactored the Product class, particularly the initialisation function, websocket connection and error handling.
    * Added functions to save application details (app ID, app markup, endpoint, API token):<br>
        * '_set_app_details'
        * '__get_app_markup'
        * '__get_app_id'
    * Improved 'sell_contract' and 'contract_proposal' functions.
* Turbo Class
    * Refactored the Turbo class.
    * Added an option to switch between the 'original' or 'revamp' version of Turbos.
* Vanilla Options
    * Added check_payout functionality and added comments for clarity.

Milestones achieved as 19/08/24:
* GUI Development
    * Completed the initial version of the GUI with the following features:
        * Buying/selling contracts
        * Changing Product type on the fly
        * Changing contract parameters
        * Running validation functions
* Documentation
    * Comprehensive documentation has been added in docstrings for most functions.
    * Documentation available in HTML (website) format for easy viewing
    * Implemented Sphinx to generate documentation
* Code Enhancements
    * Enhancements to code structure made for better compatibility with GUI
    * Reorganised file structure and enhance yaml reading function
* API related
    * Add ability to add limit orders when purchasing contracts
    * Take profit & Stop loss

Milestones achieved as 28/10/24:
* GUI Completion:
    * The GUI is now fully developed, making the tool more accessible. The interface can also serve as a quick onboarding and can demonstrate our product validation process to any new team members. Additionally, it facilitates:
      * **Comprehensive Validation:** Running all validation functions to validate areas, including bid/ask prices, barriers, etc.
      * **Account Management:** Logging into accounts and changing API endpoints on the fly.
      * **Order Execution:** Buying, selling and requesting proposals with ease.
      * **Contract Parameters:** Changing contract parameters such as stake, barriers, etc based on the current product type.
* New Data Export Feature:
    * Feature to save all validation data from each validation task when needed. Now, each session’s data can exported to an .xlsx workbook for easy record, including:
      * Endpoints & App_id
      * Validation statistics
      * Accounts used
      * Contract IDs of purchased contracts
      * All API responses

Milestones achieved as 11/11/24:
* Refactor code:
    * The GUI code has been incoporated into the mvapi.py file.
* Example Script.ipynb has been updated to show the up to date features and use, including how to enable the GUI. 

Milestones Achieved as of 19/11/24:
* requirements.txt Added & Command-Line Execution:
    * Created requirements.txt for easier dependency installation. Users can now set up the environment quickly with minimal effort using pip install -r requirements.txt.
    * Added instructions for running mvapi.py via the command line or using .command file on macOS, allowing seamless execution across different environments.

### Future functionalities (planned):
* Automate testing for other products (Turbos, Accumulators) **[Done]**
* Refactor code for easy product integration of new products **[Done]**
* Index parameters backwards engineering (If applicable)
* Convert script to a reusable library **[Done]**
* Production integration for live automated testing
* Develop GUI for ease of use for manual testing **[Done]**
* Add functionality to price from shortcode. 
