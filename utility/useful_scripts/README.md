# Useful Scripts Documentation

This documentation provides information about the utility scripts available in this folder.

## Main Scripts

### Python Scripts



#### QAbox_feed_getter.py
Python utility for retrieving and processing data feeds in the QA environment. This script:
- Retrieves real-time feed data for testing and quality assurance
- Requires user credentials from company LastPass for authentication
- Processes and formats feed data for analysis
- Supports various feed types and data formats

Usage:
1. Ensure the required Python libraries are installed:
   ```bash
   pip install psycopg2 pandas
   ```

2. Run the script using Python 3:
   ```bash
   python3 QAbox_feed_getter.py
   ```

3. Before running, add the credentials in the script. Credentials have the following format:
   ```python
   user_credentials = {
       "feed": {
           "host": "feeddb10-qa.regentmarkets.com",
           "port": "5432",
           "dbname": "feed",
           "user": "",
           "password": "",
       },
       "chronicle": {
           "host": "chronicledb-qa.regentmarkets.com",
           "port": "5432",
           "dbname": "chronicle",
           "user": "",
           "password": "",
       }
   }
   ```
   - The `password` field must be filled with the credentials obtained from LastPass. 
   - If you don't have access to the credentials on LastPass, you can request access from Amir Naser.

#### DSI_comm_jsonifier.py
Python script for converting DSI (Drift Switch Index) communications into JSON format. Useful for data standardization and API integration.


### Shell Scripts

#### restart_service.sh
Shell script for restarting various services. Useful for maintenance and troubleshooting.

Usage:
1. Make the script executable:
   ```bash
   chmod +x restart_service.sh
   ```

2. Run the script:
   ```bash
   ./restart_service.sh
   ```


#### create_accs.sh
Shell script that automates the creation of multiple trading accounts. This script is a batch wrapper around `create_account.pl`, making it easy to create several accounts in one go.

Usage:
1. Make the script executable:
   ```bash
   chmod +x create_accs.sh
   ```

2. Run the script:
   ```bash
   ./create_accs.sh
   ```

3. Modify `mail_no` as needed:
   - Set `mail_no` to `1`, `2`, or any number to add a suffix (e.g., `+1` or `+2`) to email addresses in the format `some_email+1@mail.com` for creating additional test accounts.
   - By default, `mail_no` is set to `0`, which uses the standard format without the suffix.

4. Ensure the `create_account.pl` script is in the **same directory** as this shell script for the script to execute properly.

### Perl Scripts

#### backpricing.pl
A Perl script for back-pricing financial instruments. Used to calculate historical prices for various financial products.

#### create_account.pl
Perl script for creating individual trading accounts. This script helps in setting up new trading accounts with specified param#### create_account.pl
Perl script for creating individual trading accounts. This script helps in setting up new trading accounts with specified parameters.

Usage:
```bash
perl create_account.pl some_email@mail.com Abcd1234 CR aq
```

- **`some_email@mail.com`**: The email address for the trading account.
- **`Abcd1234`**: The password for the trading account.
- **`CR`**: The broker. Use `MF` if creating an EU account.
- **`aq`**: The country code (e.g., `aq` for Antarctica, `id` for Indonesia).

Make sure to provide valid inputs for each parameter. The broker code (`CR` or `MF`) needs to be entered in uppercase.eters.

## Python Notebooks

### Feed Monitoring.ipynb
Jupyter notebook for monitoring and analyzing data feeds. This interactive notebook provides tools for:
- Visualizing feed data
- Analyzing client trading patterns
- Summarise PnL figures

## Perl Tools

### Volatility Surface Tools
Located in the `VolSurface` directory, these Perl scripts are used for volatility calculations and analysis:

#### get_vol_by_delta.pl
Retrieves volatility values based on delta parameters. Useful for:
- Delta-based volatility analysis
- Options pricing using delta parameters

#### get_vol_by_strike.pl
Calculates volatility values for specific strike prices. Used for:
- Strike-based volatility analysis
- Options pricing at specific strike levels

#### get_vol_example.pl
Example script demonstrating the usage of volatility calculation functions. Provides:
- Sample code for volatility calculations
- Usage examples for other volatility scripts

#### get_vol_smile.pl
Generates and analyzes volatility smile curves. Features:
- Volatility smile visualization
- Smile curve analysis
- Market calibration tools

## Usage Examples

### Python Scripts

#### Converting DSI Communications to JSON
```bash
python DSI_comm_jsonifier.py input_file.txt output_file.json
```

#### Using Feed Monitoring
1. Open Jupyter Notebook:
```bash
jupyter notebook "Monitoring Tools/Feed Monitoring.ipynb"
```
2. Follow the interactive cells in the notebook for feed analysis

### Perl Scripts


#### Getting Volatility Data
```bash
./VolSurface/get_vol_by_strike.pl --symbol=EURUSD --strike=1.1000
```


## Important Notes
- All scripts must be run from a QA box environment
- Ensure proper permissions are set before executing scripts
- Some scripts require specific environment variables or configuration files
- User credentials should be obtained from company LastPass
- For detailed information about each script, refer to the inline documentation within the scripts
