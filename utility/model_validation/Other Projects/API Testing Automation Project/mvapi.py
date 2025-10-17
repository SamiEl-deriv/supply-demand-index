#!/usr/bin/env python
# coding: utf-8

from IPython.display import display, clear_output
import json
import re
import websocket
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import random
import time
import pandas as pd
import os
import sys
import yaml
from datetime import datetime
from scipy.special import betainc
import tkinter as tk
from tkinter import ttk
import inspect
from tabulate import tabulate

def convert_np_types(obj) -> any:
    """
    Converts NumPy types to native Python types for serialization.

    Parameters
    ----------
    obj : any
        The object to be converted.

    Returns
    -------
    any
        The converted object, compatible with JSON serialization.

    Raises
    ------
    TypeError
        If the object type is not serializable.
    """
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    else:
        raise TypeError(f"Type {type(obj)} not serializable")
        
class MvapiGUI:
    """
    Graphical User Interface (GUI) for managing the MVAPI interactions.

    Attributes
    ----------
    master : Tk
        The main Tkinter root window.
    Product : object
        The product object associated with the GUI.
    api_token : str
        API token for WebSocket communication.
    """
    def __init__(self, master, product) -> None:
        """
        Initializes the GUI with the given master window and product.

        Parameters
        ----------
        master : Tk
            The Tkinter root window.
        product : bool or object
            The product to associate with the GUI. If False, defaults to the Accumulator instance.
        """        
        self.master = master
        master.title("MVAPI GUI")
        master.geometry("1600x1000")
        if product == False:
            self.Product = Accumulator()
        else:
            self.Product = product
        self.api_token = 0
        self.create_frames()
        self.create_api_frame()
        self.create_endpoint_frame()
        self.create_button_frame()
        self.create_product_dropdown()
        self.create_check_dropdown()
        self.create_param_frames()
        self.create_output_frame()
        self.create_export_subframe()
        self.create_status_bar()

    def create_frames(self) -> None:
        """
        Creates the left and right frames for the GUI, with scrolling support in the left frame.
        """
        # Create a canvas and scrollbar for the left frame with defined width
        left_frame_canvas = tk.Canvas(self.master, width=300)  # Set a fixed width for left frame canvas
        # Create the right frame as usual, ensuring it fills the remaining space
        self.right_frame = tk.Frame(self.master)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(0, 5), pady=5)

        # Scrollbar for left frame canvas
        left_scrollbar = tk.Scrollbar(self.master, orient="vertical", command=left_frame_canvas.yview)
        left_frame_canvas.configure(yscrollcommand=left_scrollbar.set)

        # Pack the scrollbar and left frame canvas side by side
        left_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        left_frame_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)  # Set expand to False

        # Create an internal frame for widgets within the left frame canvas
        self.left_frame = tk.Frame(left_frame_canvas)
        self.left_frame.bind(
            "<Configure>",
            lambda e: left_frame_canvas.configure(scrollregion=left_frame_canvas.bbox("all"))
        )

        # Add the frame to the canvas
        left_frame_canvas.create_window((0, 0), window=self.left_frame, anchor="nw")

        # Smooth scrolling setup
        def on_mouse_wheel(event):
            if event.num == 5 or event.delta < 0:
                left_frame_canvas.yview_scroll(1, "units")  # Scroll down
            elif event.num == 4 or event.delta > 0:
                left_frame_canvas.yview_scroll(-1, "units")  # Scroll up

        self.left_frame.bind_all("<MouseWheel>", on_mouse_wheel)   # For Windows
        self.left_frame.bind_all("<Button-4>", on_mouse_wheel)     # For Linux, scroll up
        self.left_frame.bind_all("<Button-5>", on_mouse_wheel)     # For Linux, scroll down

    def create_api_frame(self) -> None:
        """
        Creates the API Token frame where tokens can be entered, saved, or deleted.
        """        
        api_frame = tk.Frame(self.left_frame, height=100)
        api_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(api_frame, text="API Token:").pack(anchor=tk.W)
        self.api_entry = tk.Entry(api_frame, width=30)
        self.api_entry.pack(anchor=tk.W)

        button_container = tk.Frame(api_frame)
        button_container.pack(anchor=tk.W, pady=5)

        tk.Button(button_container, text="Save", command=self.set_api_token).pack(side=tk.LEFT, padx=5)
        tk.Button(button_container, text="Delete", command=self.delete_api_token).pack(side=tk.LEFT)

    def create_endpoint_frame(self) -> None:
        """
        Creates a frame for managing the WebSocket endpoint.
        """        
        endpoint_frame = tk.Frame(self.left_frame, height=100)
        endpoint_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(endpoint_frame, text="Endpoint:").pack(anchor=tk.W)
        self.endpoint_entry = tk.Entry(endpoint_frame, width=30)
        self.endpoint_entry.pack(anchor=tk.W)

        button_container = tk.Frame(endpoint_frame)
        button_container.pack(anchor=tk.W, pady=5)

        tk.Button(button_container, text="Save", command=self.set_endpoint).pack(side=tk.LEFT, padx=5)
        
    def create_button_frame(self) -> None:
        """
        Creates buttons for buying, proposing, and managing contracts.
        """        
        button_frame = tk.Frame(self.left_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=(2,2))

        tk.Button(button_frame, text="Buy Contract", command=self.buy_contract).pack(fill=tk.X, pady=5)
        tk.Button(button_frame, text="Price Proposal", command=self.price_proposal).pack(fill=tk.X, pady=5)
        tk.Button(button_frame, text="Sell Contract", command=self.sell_contract).pack(fill=tk.X, pady=5)
        tk.Button(button_frame, text="Sell All Contracts", command=self.sell_all).pack(fill=tk.X, pady=5)

    def create_product_dropdown(self) -> None:
        """
        Creates a dropdown to select different product types.
        """
        product_types = ['Accumulator', 'Turbo', 'Vanilla', 'RiseFall', 'Multiplier']
        tk.Label(self.left_frame, text="Available Product Types").pack()
        self.product_box = ttk.Combobox(self.left_frame, values=product_types)
        self.product_box.pack(fill=tk.X, padx=10)
        self.product_box.current(0)
        self.product_box.bind('<<ComboboxSelected>>', self.change_product)

    def create_check_dropdown(self) -> None:
        """
        Creates a dropdown to select available check functions for the product.
        """        
        tk.Label(self.left_frame, text="Available Check Functions").pack()
        self.check_box = ttk.Combobox(self.left_frame, values=self.get_check_options())
        self.check_box.pack(fill=tk.X, padx=10)
        self.check_box.current(0)

    def create_param_frames(self) -> None:
        """
        Creates frames for user input parameters and check function parameters.
        """        
        self.check_param_frame = tk.Frame(self.left_frame, bd=2, relief=tk.SOLID)
        self.check_param_frame.pack(fill=tk.BOTH, pady=(10,0))

        self.param_frame = tk.Frame(self.left_frame, bd=2, relief=tk.SOLID)
        self.param_frame.pack(expand=True, fill=tk.BOTH, pady=(10,0))

        self.update_check_params(self.get_check_input_params())
        self.update_user_input_params(self.get_user_input_params())

    def create_output_frame(self) -> None:
        """
        Creates a text output frame with scrollable functionality.
        """        
        output_frame = tk.Frame(self.right_frame, bd=2, relief=tk.SOLID)
        output_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        tk.Label(output_frame, text="Output").pack(anchor=tk.N, padx=10, pady=5)
        self.output_text = tk.Text(output_frame, font=("Menlo", 14))
        self.output_text.pack(expand=True, fill=tk.BOTH)
        
    def create_export_subframe(self) -> None:
        """
        Creates a subframe for exporting data to a file.
        """        
        # Create a subframe within the output frame
        export_frame = tk.Frame(self.right_frame, bd=2, relief=tk.SOLID)
        export_frame.pack(side=tk.BOTTOM, fill=tk.X)#, expand=True)

        # Input for file name
        tk.Label(export_frame, text="File Name:").pack(side=tk.LEFT, padx=5)
        self.filename_entry = tk.Entry(export_frame, width=20)
        self.filename_entry.pack(side=tk.LEFT, padx=5)

        # Dropdown for timestamp option
        tk.Label(export_frame, text="Save Timestamp:").pack(side=tk.LEFT, padx=5)
        self.timestamp_var = tk.StringVar()
        timestamp_dropdown = ttk.Combobox(export_frame, textvariable=self.timestamp_var, values=["Yes", "No"], width=10)
        timestamp_dropdown.pack(side=tk.LEFT, padx=5)
        timestamp_dropdown.current(1)  # Default to "No"

        # Button for exporting data
        export_button = tk.Button(export_frame, text="Export Data", command=self.export_data_action)
        export_button.pack(side=tk.LEFT, padx=10)
        
    def create_status_bar(self) -> None:
        """
        Creates a status bar at the bottom of the application to display API-related statuses.
        """        
        status_frame = tk.Frame(self.left_frame, height=20)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0,5))
        tk.Label(status_frame, text="Status: ").pack(side=tk.LEFT, anchor=tk.W, pady=2)
        self.status_output = tk.Label(status_frame, text="API Token not set.", anchor='w')
        self.status_output.pack(side=tk.LEFT, fill=tk.X)

    def set_api_token(self) -> None:
        """
        Saves the given API token and attempts authorization.
        """        
        self.api_token = self.api_entry.get()
        self.Product.api_token = self.api_token
        self.update_status(f"API Token set to: {self.api_token}.")
        self.update_output(json.dumps(self.Product.last_msg, indent=1))

    def set_endpoint(self) -> None:
        """
        Saves and updates the WebSocket endpoint.
        """        
        self.endpoint = self.endpoint_entry.get()
        self.Product.endpoint = self.endpoint
        self.update_status(f"Endpoint changed.")
        self.update_output(json.dumps(self.Product.last_msg, indent=1))
        
    def delete_api_token(self) -> None:
        """
        Deletes the currently saved API token and resets connection.
        """        
        self.api_token = 0
        self.Product.api_token = self.api_token
        self.Product.create_ws()
        self.update_status("API Token deleted.")
        self.update_output("API token deleted.\n")

    def update_status(self, status) -> None:
        """
        Updates the text in the status bar.

        Parameters
        ----------
        status : str
            The status message to display.
        """        
        self.status_output.config(text=status)
        self.master.update()

    def update_output(self, string, clear=True) -> None:
        """
        Updates the output frame text.

        Parameters
        ----------
        string : str
            The string to display in the output frame.
        clear : bool, optional
            Whether to clear existing text (default is True).
        """        
        if clear:
            self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, string)
        self.master.update()

    def buy_contract(self, **kwargs) -> None:
        """
        Initiates a contract purchase and updates output with contract details.
        """        
        self.update_status("Attempting to buy contract.")
        temp_flag = self.Product.buy_contract(**kwargs)
        self.update_output(json.dumps(self.Product.last_msg, indent=1))
        self.update_status("Bought contract." if temp_flag else "Failed to buy.")
        
    def price_proposal(self, **kwargs) -> None:
        """
        Requests a price proposal and displays the response.
        """        
        self.update_status("Attempting to get price proposal.")
        temp_flag = self.Product.price_proposal(**kwargs)
        self.update_output(json.dumps(self.Product.last_msg, indent=1))
        self.update_status("Got price proposal." if temp_flag else "Failed to get price proposal.")
        
    def sell_contract(self, **kwargs) -> None:
        """
        Attempts to sell a specific contract and updates the output.
        """        
        self.update_status("Attempting to sell contract.")
        temp_flag = self.Product.sell_contract(**kwargs)
        self.update_output(json.dumps(self.Product.last_msg, indent=1))
        self.update_status("Sold contract." if temp_flag else "Failed to sell.")

    def sell_all(self, **kwargs) -> None:
        """
        Attempts to sell all currently active contracts.
        """        
        self.update_status("Attempting to sell all contracts.")
        temp_flag = self.Product.sell_all_contracts(**kwargs)
        index = len(self.Product.GUI_data)
        self.update_output('\n' + json.dumps(self.Product.GUI_data[index:], indent=1, default=convert_np_types), clear=True)
        self.update_status("Attempt to sell all contracts ran.")

    def change_product(self, event=None) -> None:
        """
        Changes the active product in the GUI.
        """        
        product_type = self.product_box.get()
        if product_type != self.Product.species:
            self.update_status(('Changing', self.Product.species, 'to', product_type))
            get_class = getattr(sys.modules[__name__], product_type)
            self.Product = get_class(**self.Product.__dict__)
            self.update_check_options()
            self.update_status(f"Product type changed to {product_type}.")
            self.update_output(f"Current product type is {self.Product.species}.")

    def get_check_options(self) -> list:
        """
        Retrieves a list of available check functions for the current product.
        """        
        return [s for s in dir(self.Product) if s.startswith('check_')]

    def update_check_options(self, event=None) -> None:
        """
        Updates the dropdown with available check functions for the selected product.
        """        
        self.check_box['values'] = self.get_check_options()
        self.check_box.current(0)
        self.update_check_params(self.get_check_input_params())
        self.update_user_input_params(self.get_user_input_params())

    def get_check_input_params(self) -> list:
        """
        Retrieves the input parameters for the currently selected check function.
        """
        check_input_params = list(inspect.signature(getattr(self.Product, self.check_box.get())).parameters.keys())
        if 'kwargs' in check_input_params:
            check_input_params.remove('kwargs')
        return check_input_params

    def get_user_input_params(self) -> list:
        """
        Retrieves user parameters for the active product.
        """        
        return self.Product.user_input_params

    def update_check_params(self, new_options) -> None:
        """
        Updates the displayed parameters for the selected check function.
        """        
        for widget in self.check_param_frame.winfo_children():
            widget.destroy()
        for opt in new_options:
            tk.Label(self.check_param_frame, text=f"{opt}:").pack(anchor=tk.W, padx=10, pady=(5, 0))
            tk.Entry(self.check_param_frame).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(self.check_param_frame, text="Run Check", command=self.run_check).pack(pady=(5,5))

    def update_user_input_params(self, new_options) -> None:
        """
        Updates the displayed parameters for user input settings.
        """        
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        for opt in new_options:
            tk.Label(self.param_frame, text=f"{opt}:").pack(anchor=tk.W, padx=10, pady=(5, 0))
            tk.Entry(self.param_frame).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(self.param_frame, text="Update Parameters", command=self.set_user_input_params).pack(pady=(5,5))

    def process_params(self, params_dict) -> dict:
        """
        Processes the input parameters and converts values to the correct types.

        Parameters
        ----------
        params_dict : dict
            The dictionary containing parameter names and values.

        Returns
        -------
        dict
            Processed parameters with converted types.
        """        
        temp_dict = {}
        for key, value in params_dict.items():
            if value != '':
                if key in ['runs', 'duration', 'show']:
                    temp_dict[key] = int(value)
                elif key in ['stake', 'take_profit', 'stop_loss', 'strike', 'growth_rate']:
                    temp_dict[key] = float(value)
                else:
                    temp_dict[key] = value
        return temp_dict

    def set_check_params(self) -> dict:
        """
        Collects and processes parameter values for the selected check function.
        """        
        params_dict = {}
        for widget in self.check_param_frame.winfo_children():
            if isinstance(widget, tk.Label):
                option_name = widget.cget("text").strip(":")
            elif isinstance(widget, tk.Entry):
                option_value = widget.get()
                params_dict[option_name] = option_value
        return self.process_params(params_dict)

    def set_user_input_params(self) -> None:
        """
        Sets user-specific parameters for the active product.
        """        
        params_dict = {}
        for widget in self.param_frame.winfo_children():
            if isinstance(widget, tk.Label):
                option_name = widget.cget("text").strip(":")
            elif isinstance(widget, tk.Entry):
                option_value = widget.get()
                params_dict[option_name] = option_value
        params_dict = self.process_params(params_dict)
        self.Product._set_attributes(**params_dict)
        self.update_output('Contract Parameters set to:\n')
        self.update_output(str(params_dict), clear=0)

    def run_check(self) -> None:
        """
        Executes the selected check function with the provided parameters and displays results.
        """        
        check_type = self.check_box.get()
        self.update_status(('Running', check_type))
        index = len(self.Product.GUI_data)
        results_df = getattr(self.Product, check_type)(**self.set_check_params())
        self.update_output(results_df.to_markdown())
        self.update_output('\n' + json.dumps(self.Product.GUI_data[index:], indent=1, default=convert_np_types), clear=False)
        self.update_status((check_type, 'complete'))
        
    def export_data_action(self) -> None:
        """
        Initiates data export to an Excel file using the specified file name and timestamp settings.
        """        
        # Get values from input fields
        filename = self.filename_entry.get()
        timestamp = self.timestamp_var.get()
        # Call export_data function from the Product class
        self.update_status(('Exporting data'))
        file_path = self.Product.export_data(file_name=filename, save_timestamp=(True if timestamp == "Yes" else False))
        self.update_output('Data exported as ' + file_path)     
        self.update_status(('Data exported as ' + file_path))

def run_gui(product=False) -> object:
    """
    Launches the graphical user interface (GUI) for the application.

    Parameters
    ----------
    product : bool, optional
        Specifies whether to initialize the GUI with a product (default is False).

    Returns
    -------
    Product
        The product object created or modified through the GUI.
    """    
    root = tk.Tk()
    gui = MvapiGUI(master=root, product=product)
    root.mainloop()
    return gui.Product
    

# Get the directory where the current script is located
script_dir = os.path.dirname(__file__)

def load_yaml_file(filename) -> dict:
    """
    Load a YAML file and return its contents.

    This function reads a YAML file located in the same directory as the script and returns its contents
    as a dictionary. It handles potential errors such as YAML parsing issues or file not found exceptions.

    Parameters
    ----------
    filename : str
        The name of the YAML file to be loaded. The file should be located in the same directory as the script.

    Returns
    -------
    dict
        The contents of the YAML file as a dictionary. Returns `None` if there is an error loading the file.

    Raises
    ------
    yaml.YAMLError
        If there is an error parsing the YAML file.
    FileNotFoundError
        If the specified file does not exist.
    Exception
        For any other exceptions that occur during file loading.

    Examples
    --------
    >>> symbols = load_yaml_file("SYMBOLS.yaml")
    >>> BO = load_yaml_file("BO.yaml")
    >>> default_loss_probs = load_yaml_file("LOSS_PROBS.yaml")
    >>> contract_types = load_yaml_file("CONTRACT_TYPES.yaml")

    Notes
    -----
    The function prints error messages to the standard output if it encounters issues while loading or parsing the file.
    """
    filepath = os.path.join(script_dir, filename)
    try:
        with open(filepath) as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(f"YAML error in {filename}: {e}")
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error loading file {filename}: {e}")

symbols = load_yaml_file("SYMBOLS.yaml")
BO = load_yaml_file("BO.yaml")
default_loss_probs = load_yaml_file("LOSS_PROBS.yaml")
contract_types = load_yaml_file("CONTRACT_TYPES.yaml")

def accu_tick_size_barrier_builder(loss_probs=default_loss_probs) -> None:
    """
    Builds the tick size barriers for each growth rate for each symbol based on their loss probabilities.

    Parameters
    ----------
    loss_probs : dict, optional
        A dictionary with keys as growth rates and values as corresponding loss probabilities.
        Defaults to `default_loss_probs`.

    Returns
    -------
    None

    Notes
    -----
    Run at the start to build the tick size barrier for each growth rate for each symbol based on preset default values.
    
    The function updates the tick size barrier for each growth rate for each symbol in the 'symbols' dictionary
    by solving for the barrier for each growth rate using the given loss probabilities. 

    Examples
    --------
    >>> loss_probs = {'0.01': 0.01, '0.02': 0.02, '0.03': 0.03, '0.04': 0.04, '0.05': 0.05}
    >>> accu_tick_size_barrier_builder(loss_probs)
    """

    def is_vol(symbol):
        return 1 if ('HZ' in symbol or 'R_' in symbol) else 0 

    for symbol in symbols.keys():
        if is_vol(symbol):
            symbols[symbol]['accu'] = {'barrier' : {}}

    def gbm(barrier, loss_prob, sigma, symbol) -> float:
        t = (2 if 'R' in symbol else 1)/365/86400
        mu = -0.5*sigma**2*t
        sd = sigma**2*t
        prob_b1 = 1 - norm.cdf((np.log(1+barrier)-mu)/np.sqrt(sd))
        prob_b2 = norm.cdf((np.log(1-barrier)-mu)/np.sqrt(sd))
        return (prob_b1 + prob_b2 - loss_prob)

    for growth_rate in loss_probs.keys():
        loss_prob = loss_probs[growth_rate]
        for symbol in symbols.keys():
            if is_vol(symbol):
                sigma = symbols[symbol]['sigma']
                symbols[symbol]['accu']['barrier'][growth_rate] = fsolve(gbm, 0, args = (loss_prob, sigma, symbol))[0]

accu_tick_size_barrier_builder()

def insert_dict(item, key, _dict) -> None:
    """
    Insert or update a dictionary into the key of the given item.

    This function attempts to update the specified key of the provided item with the 
    key-value pairs from the provided dictionary. If the key does not exist in the item, 
    it initializes the key as an empty dictionary before performing the update.

    Parameters
    ----------
    item : dict
        The dictionary in which the key-value pairs are to be inserted or updated.
    key : str
        The key within the item dictionary to be updated with the provided dictionary.
    _dict : dict
        The dictionary containing key-value pairs to be inserted or updated into the item.

    Returns
    -------
    None

    Examples
    --------
    >>> item = {'a': {}}
    >>> insert_dict(item, 'b', {'y': 2})
    >>> print(item)
    {'a': {}, 'b': {'y': 2}}

    >>> item = {'a': {}, 'b': {}}
    >>> insert_dict(item, 'b', {'y': 2})
    >>> print(item)
    {'a': {}, 'b': {'y': 2}}
    """

    try: 
        item[key].update(_dict)
    except Exception as e:
        item[key] = {}
        try: 
            item[key].update(_dict)
        except Exception as e:
             item[key] = _dict

def dict_to_df(_dict, title) -> pd.DataFrame:
    """
    Convert a dictionary to a pandas DataFrame.

    This function takes a dictionary and converts it to a pandas DataFrame. The dictionary's keys 
    become a 'Category' column in the DataFrame, and the associated values become other columns. 
    If the dictionary cannot be converted, it creates a DataFrame with the dictionary's content.

    Parameters
    ----------
    _dict : dict
        The dictionary to be converted to a DataFrame.
    title : str
        The title to be printed alongside the DataFrame for identification.

    Returns
    -------
    pd.DataFrame
        The resulting DataFrame created from the dictionary.

    Examples
    --------
    >>> _dict = {'A': {'value1': 10, 'value2': 20}, 'B': {'value1': 30, 'value2': 40}}
    >>> df = dict_to_df(_dict, 'Example Title')
    Stats for Example Title check
    >>> display(df)
    Category  value1  value2
    0        A      10      20
    1        B      30      40
    """

    try: 
        flattened_data = [{**{'Category': sub_key}, **values} for sub_key, values in _dict.items()]
        df = pd.DataFrame(flattened_data)
    except Exception as e:
        df = pd.DataFrame([_dict])
    print('Stats for', title, 'check')
    display(df)
    return df

def get_sigma(symbol) -> float:
    """
    Get the sigma value for a given symbol.

    This function retrieves the 'sigma' value associated with the provided symbol 
    from a predefined symbols dictionary.

    Parameters
    ----------
    symbol : str
        The symbol for which the sigma value is to be retrieved.

    Returns
    -------
    float
        The sigma value associated with the given symbol.

    Examples
    --------
    >>> get_sigma('1HZ75V')
    0.75
    """
    return symbols[symbol]['sigma']

def get_stake(prop) -> float:
    """
    Get the stake amount from a given proposal.

    This function calculates the stake amount from the provided proposal by subtracting 
    the app markup amount from the proposal's requested amount. If the app markup amount 
    is not available, it returns the requested amount.

    Parameters
    ----------
    prop : dict
        The proposal from which the stake amount is to be retrieved.

    Returns
    -------
    float
        The calculated stake amount from the given proposal.

    Examples
    --------
    >>> prop = {
    ...     'echo_req': {'amount': 100},
    ...     'proposal': {'contract_details': {'app_markup_amount': '5'}}
    ... }
    >>> get_stake(prop)
    95.0

    >>> prop = {
    ...     'echo_req': {'amount': 100}
    ... }
    >>> get_stake(prop)
    100.0
    """
    if prop['msg_type'] == 'proposal':
        try:
            stake = prop['echo_req']['amount'] - float(prop['proposal']['contract_details']['app_markup_amount'])
        except Exception as e:
            stake = prop['echo_req']['amount']
    else:
        try:
            stake = float(prop['proposal_open_contract']['buy_price']) - float(prop['proposal_open_contract']['contract_details']['app_markup_amount'])
        except Exception as e:
            stake = float(prop['proposal_open_contract']['buy_price'])
        try:
            stake = stake - prop['proposal_open_contract']['cancellation']['ask_price']
        except Exception as e:
            stake = stake
    return stake

def get_currency_precision(prop) -> int:
    """
    Retrieves the currency precision for the given proposal.

    Parameters
    ----------
    prop : dict
        The proposal object containing currency information.

    Returns
    -------
    int
        The currency precision for the specified proposal's currency.
    """    
    return BO['currency_precision'][prop['proposal_open_contract']['currency']]

def roundsf(number) -> float:
    """
    Round a number to retain only the first non-zero digit.

    This function takes a number, retains only the first non-zero digit, and 
    replaces all following digits with zeros.

    Parameters
    ----------
    number : float
        The number to be rounded.

    Returns
    -------
    float
        The rounded numeric value.

    Examples
    --------
    >>> roundsf(123.456)
    100.0

    >>> roundsf(0.00456)
    0.004
    """
    number = "{:.15f}".format(number)
    non_zero_index = re.search('[^0.]', number).start()        
    return float(number[:non_zero_index+1] + re.sub('[0-9]', '0', number[non_zero_index+1:]))

def roundup(num, prec=0) -> float:
    """
    Round up a number to the specified precision.

    This function rounds up a number to the specified number of decimal places.

    Parameters
    ----------
    num : float
        The number to be rounded up.
    prec : int
        The number of decimal places to round up to.

    Returns
    -------
    float
        The number rounded up to the specified precision.

    Examples
    --------
    >>> roundup(123.456, 2)
    123.46

    >>> roundup(0.00456, 3)
    0.005
    """
    return np.ceil(round((num * 10**prec),15)) / (10**prec)

def rounddown(num, prec=0) -> float:
    """
    Round down a number to the specified precision.

    This function rounds down a number to the specified number of decimal places.

    Parameters
    ----------
    num : float
        The number to be rounded down.
    prec : int
        The number of decimal places to round down to.

    Returns
    -------
    float
        The number rounded down to the specified precision.

    Examples
    --------
    >>> rounddown(123.456, 2)
    123.45

    >>> rounddown(0.00456, 3)
    0.004
    """
    return np.floor(round((num * 10**prec),15)) / (10**prec)

def set_limit_orders(obj, take_profit=None, stop_loss=None) -> None:
    """
    Sets the take profit and stop loss limits for the given object.

    This function updates the take profit and stop loss attributes of the object if they are
    specified and valid to be set according to the contract types.

    Parameters
    ----------
    obj : object
        The object whose limit orders are to be set.

    take_profit : float, int, optional
        The take profit value to be set. Defaults to None.

    stop_loss : float, int, optional
        The stop loss value to be set. Defaults to None.

    Returns
    -------
    None
    """    
    if contract_types[obj.species]['take_profit'] and type(take_profit) in [float,int]:
        obj.take_profit = round(take_profit, BO['currency_precision'][obj.currency])
    if contract_types[obj.species]['stop_loss'] and type(stop_loss) in [float,int]:
        obj.stop_loss = round(stop_loss, BO['currency_precision'][obj.currency])

def set_mult_cancellation(obj, cancellation) -> None: 
    """
    Sets the cancellation value for a multiplier contract.

    Parameters
    ----------
    obj : object
        The contract object to update.
    cancellation : any
        The cancellation value to set.

    Returns
    -------
    None

    Notes
    -----
    - If the symbol contains 'stp', the cancellation is set to 0.
    - If the value is invalid, it attempts to append 'm' (e.g., "10m").
    """    
    if 'stp' in obj.symbol:
        obj.cancellation = 0
    elif cancellation in BO['multipliers']['cancellation']:
        obj.cancellation = cancellation
    else:
        try: 
            cancellation = str(cancellation) + 'm'
            obj.cancellation = cancellation if cancellation in BO['multipliers']['cancellation'] else 0
        except Exception as e:
            obj.cancellation = 0

class Product:
    """
    A base class to represent a product for trading contracts through DerivAPI via WebSocket.

    Attributes
    ----------
    api_token : str
        API token for authentication.
    stake : int
        The amount to stake in each trade.
    symbol : str
        The underlying symbol of the product.
    messages : list
        List to store messages received from the WebSocket.
    endpoint : str
        The WebSocket endpoint.
    retry : int
        Counter for retry attempts when sending/receiving messages.
    contract_ids : list
        List of contract IDs.
    ws : WebSocket
        WebSocket connection object.

    Methods
    -------
    __setattr__() : None
        Set the value of an attribute and trigger corresponding actions based on the attribute.
    _set_attributes(**kwargs) : None
        Set multiple attributes on the object using keyword arguments.
    _set_app_details() : None
        Saves application details like endpoint and API token.
    _get_app_markup() : float
        Retrieves the application markup percentage.
    _get_app_id(string) : int
        Extracts the application ID from the endpoint URL.
    _proposal() : str
        Generates a proposal based on the current proposal type.
    _send() : dict
        Sends a proposal to the WebSocket and handles the response.
    send_message(message) : dict
        Sends a user-defined JSON message to the WebSocket.
    stream() : int
        Streams data from the WebSocket.
    price_proposal(random=False, buy=False, take_profit=0, stop_loss=0, **kwargs) : bool
        Gets a price proposal.
    buy_contract(runs=1, random=False, **kwargs) : None
        Attempts to buy a contract.
    contract_proposal(contract_id=None, subscribe=0) : None
        Gets contract details and optionally subscribes to updates.
    get_all_poc() : None
        Gets all proposal open contracts from saved contract IDs.
    tick_history(symbol=False, count=1, end='latest') : dict
        Retrieves historical tick data.
    _check_connection() : None
        Checks the WebSocket connection and reopens it if closed.
    _check_closed() : int
        Checks if a contract is closed.
    sell_contract(contract_id=None) : None
        Attempts to sell a contract.
    sell_all_contracts() : None
        Attempts to sell all contracts.
    create_ws() : None
        Creates a new WebSocket connection.
    authorise() : int
        Authorises the WebSocket connection.
    _recv_msg_handler() : int
        Handles messages received from the WebSocket.
    """

    
    def __init__(self, **kwargs) -> None:
        """
        Initializes a new instance of the Product class.

        This constructor sets the initial attributes for the product, including default values 
        for stake, symbol, endpoint, and others. It also creates a WebSocket connection and 
        authorizes it if an API token is provided. Additionally, it sets application details.

        Parameters
        ----------
        **kwargs : dict
            Additional attributes to be set for the instance.

        Attributes
        ----------
        currency : str
            The currency of the contract.
        stake : int
            The amount to stake in each trade. Defaults to 10.
        symbol : str
            The symbol of the product. Defaults to '1HZ100V'.
        take_profit : int
            The take profit amount in each trade. Defaults to 0.
        stop_loss : int
            The stop loss amount in each trade. Defaults to 0.            
        messages : list
            List to store messages received from the WebSocket. Initialized as an empty list.
        endpoint : str
            The WebSocket endpoint. Defaults to "wss://blue.derivws.com/websockets/v3?app_id=16929".
        api_token : int
            API token for authentication. Defaults to 0.
        retry : int
            Counter for retry attempts when sending/receiving messages. Initialized to 0.
        contract_ids : list
            List of contract IDs. Initialized as an empty list.
        stats : dict
            Dictionary to store statistics. Initialized as an empty dictionary.
        app_details : dict
            Dictionary to store application details. Initialized as an empty dictionary.
        """
        
        self.retry = 0
        self.currency = 'USD'
        self.stake = 10
        self.symbol = '1HZ100V'
        self.take_profit = 0
        self.stop_loss = 0
        self.messages = []
        self.endpoint = "wss://blue.derivws.com/websockets/v3?app_id=16929"
        self.api_token = 0 
        self.contract_ids = []
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.create_ws()
        self.stats = {}
        self.GUI_data = []
        self._set_attributes(**kwargs)
        # self._set_app_details()
    
    def export_data(self, file_name=False, save_timestamp=False) -> str:
        """
        Exports the class data to an Excel file.

        This method collects various data such as details, login information, contract IDs, 
        and messages, and stores them into separate sheets in an Excel file. The file can 
        include a timestamp in the filename if specified.

        Parameters
        ----------
        file_name : str, optional
            The base name for the output file. Defaults to 'mvapi_output' if not specified.
        save_timestamp : bool, optional
            Whether to append the current timestamp to the filename. Defaults to False.

        Returns
        -------
        str
            The file path of the generated Excel file.

        Notes
        -----
        - The method creates separate Excel sheets for:
            - General details (`endpoint`, `species`).
            - Login information (`loginid`, `email`).
            - Contract IDs.
            - WebSocket messages.
            - Statistics data (stored with the original keys from `self.stats`).
        - If `save_timestamp` is True, the file name will include a timestamp.

        Examples
        --------
        >>> obj = SomeClass()
        >>> obj.export_data(file_name="output", save_timestamp=True)
        'output_2023-10-05_14-30-45.xlsx'
        """        
        timestamp = datetime.now().strftime('_%Y-%m-%d_%H-%M-%S') if save_timestamp else ''
        file_name = file_name if file_name else 'mvapi_output'
        logins = {}
        for msg in self.messages:
            if 'authorize' in msg:
                logins[msg['authorize']['loginid']] = msg['authorize']['email']
        details = {'endpoint': self.endpoint, 'species': self.species}
        df_details = pd.DataFrame(list(details.items()), columns=['area', 'detail'])
        df_logins = pd.DataFrame(list(logins.items()), columns=['loginid', 'email'])
        df_contract_ids = pd.DataFrame({'contract_ids': self.contract_ids})
        df_messages = pd.DataFrame({'messages': self.messages})
        file_path = file_name + timestamp + '.xlsx'
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_details.to_excel(writer, sheet_name='details', index=False)
            df_logins.to_excel(writer, sheet_name='accounts', index=False)
            df_contract_ids.to_excel(writer, sheet_name='contract_ids', index=False)
            for key in self.stats.keys():
                dict_to_df(self.stats[key],key).to_excel(writer, sheet_name=key, index=False)
            df_messages.to_excel(writer, sheet_name='ws_responses', index=False)
        return file_path

    def __setattr__(self, key: str, value) -> None:
        """
        Set the value of an attribute and trigger corresponding actions based on the attribute.

        Parameters
        ----------
        key : str
            The name of the attribute to set.
        value : any
            The value to assign to the attribute.

        Returns
        -------
        None

        Notes
        -----
        - If `key` is 'endpoint', the WebSocket connection is created.
        - If `key` is 'api_token', the authorization process is initiated.
        """
        super().__setattr__(key, value)
        match key:
            case 'endpoint':
                # self._set_app_details(self)
                self.create_ws()
            case 'api_token': 
                if value:
                    # self._set_app_details(self)
                    self.authorise()

    def _set_attributes(self, **kwargs) -> None:
        """
        Set multiple attributes on the object using keyword arguments and trigger corresponding actions based on the attribute.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of key-value pairs where each key is an attribute name and each value is the attribute value to set.

        Returns
        -------
        None

        Notes
        -----
        - If `key` is 'take_profit', the take profit limit order is set.
        - If `key` is 'stop_loss', the stop loss limit order is set.
        """        
        for key, value in kwargs.items():
            match key:
                case 'take_profit':
                    set_limit_orders(self, take_profit=value) 
                case 'stop_loss':
                    set_limit_orders(self, stop_loss=value) 
                case 'cancellation':
                    set_mult_cancellation(self, cancellation=value)
                case _:
                    setattr(self, key, value)

    # def _set_app_details(self) -> None:
    #     """
    #     Retrieves application details such as endpoint, API token, app ID, and app markup percentage.

    #     This method updates the `app_details` attribute with the current endpoint and API token. 
    #     It also retrieves and adds the application ID and markup percentage to the `app_details`.

    #     Returns
    #     -------
    #     None
    #     """
    #     self.app_details = {
    #         'endpoint': self.endpoint,
    #         'api_token': self.api_token
    #     }
    #     self.app_details['app_id'] = self._get_app_id(self.endpoint)
    #     self.app_details['app_markup_percentage'] = self._get_app_markup()

    # def _get_app_markup(self) -> float:
    #     # Not feasible
    #     """
    #     Retrieves the application markup percentage.

    #     This method sends a request to get the application markup percentage after authorizing the WebSocket connection. 
    #     If an error occurs or the markup percentage is not available, it returns 0.

    #     Returns
    #     -------
    #     float
    #         The application markup percentage. Defaults to 0 if not retrievable.
    #     """
    #     if self.authorise():
    #         temp_message = {
    #         "app_get": self.app_details['app_id']
    #         }   
    #         self.send_message(temp_message)
            
    #     try:
    #         app_markup_percentage = self.last_msg['app_get']['app_markup_percentage']
    #     except Exception as e:
    #         app_markup_percentage = 0
    #     return app_markup_percentage

    # def _get_app_id(self, string) -> int:
    #     # Not feasible
    #     """
    #     Extracts the application ID from the endpoint URL.

    #     This method parses the `app_id` from the endpoint URL string. 
    #     If parsing fails, it returns 0.

    #     Parameters
    #     ----------
    #     string : str
    #         The endpoint URL from which to extract the application ID.

    #     Returns
    #     -------
    #     int
    #         The extracted application ID. Defaults to 0 if not found.
    #     """
    #     try:
    #         app_id = int(re.search(r"app_id=(\d+)", string).group(1))
    #     except Exception as e:
    #         app_id = 0
    #     return app_id

    def _proposal(self) -> str:
        """
        Generates a proposal based on the current proposal type.

        This method creates a JSON proposal based on the value of `proposal_type`. 
        It handles different proposal types such as buying a contract, 
        requesting a proposal, selling a contract, or raises an error if the type is not implemented.

        Returns
        -------
        str
            The JSON string representation of the proposal.

        Raises
        ------
        NotImplementedError
            If the `proposal_type` is 'price_proposal', indicating that this method should be used via a subclass.
        """
        proposal = {}
        if self.proposal_type == 'buy_contract':
            proposal = {
                "buy": self.proposal_id,
                "price": 1000000
            }
        elif self.proposal_type == 'contract_proposal':
            proposal = {
                "proposal_open_contract": 1,
                "contract_id": self.contract_id
            }
            if self.subscribe == 1:
                proposal["subscribe"] = 1
        elif self.proposal_type == 'sell_contract':
            proposal = {
                "sell": self.contract_id,  
                "price": 0
            }
        # To ensure this class isn't used to do price proposals without a defined contract type
        elif self.proposal_type == 'price_proposal':
            raise NotImplementedError('Use a Subclass to use this method')
                
        return json.dumps(proposal)

    def _send(self) -> bool:
        """
        Sends the current proposal to the WebSocket and handles the response.

        This method ensures the WebSocket connection is active, sends the proposal created by `_proposal()`, 
        and processes the response received from the WebSocket.

        Returns
        -------
        bool
            `True` if the message was successfully sent and a response was handled, otherwise `False`.
        """
        self._check_connection()
        self.ws.send(self._proposal())
        return self._recv_msg_handler()
        
    def send_message(self, message) -> dict:
        """
        Sends a user-defined JSON message to the WebSocket and returns the response.

        This method ensures the WebSocket connection is active, sends the provided message, 
        and processes the response. If an exception occurs, it retries sending the message.

        Parameters
        ----------
        message : dict
            The JSON message to send to the WebSocket.

        Returns
        -------
        dict
            The response received from the WebSocket.
        """
        self._check_connection()
        self.ws.send(json.dumps(message))
        try:
            response = json.loads(self.ws.recv())
        except Exception as e:
            response = self.send_message(message)
        self.last_msg = response
        self.messages.append(self.last_msg)
        return response
    
    def stream(self) -> int:
        """
        Streams data from the WebSocket.

        This method sends a proposal to the WebSocket and processes the incoming messages. 
        It handles the subscription to updates and stops streaming if the contract is closed or 
        a keyboard interrupt is detected.

        Returns
        -------
        int
            Returns 1 upon successful completion of the streaming process.
        """
        self._check_connection()
        self.ws.send(self._proposal())
        self._recv_msg_handler()
        if self._check_closed():
            print('Contract cannot be sold')
            self.subscribe = 0
            self._send()
        else:
            self.subscription_id = self.last_msg['subscription']['id']
            try:
                while True:
                    clear_output(wait = True)
                    self._recv_msg_handler()
                    display(self.last_msg)
                    print('Keyboard Interrupt to stop streaming')
                    if self._check_closed():
                        break
            except KeyboardInterrupt:
                pass
            self.ws.send(json.dumps({"forget": self.subscription_id}))
            self._recv_msg_handler()
        return 1
        
    def price_proposal(self, random=False, buy=False, **kwargs) -> bool:
        """
        Requests a price proposal from the WebSocket.

        This method sets the proposal type to 'price_proposal' and updates attributes based on 
        the provided arguments. It sends the proposal and optionally displays the response if 
        the `buy` parameter is `False`.

        Parameters
        ----------
        random : bool, optional
            Whether to randomize attributes before sending the proposal. Defaults to `False`.
        buy : bool, optional
            Whether to buy the contract immediately after receiving the proposal. Defaults to `False`.
        take_profit : float, optional
            The take profit value for the proposal. Defaults to 0.
        stop_loss : float, optional
            The stop loss value for the proposal. Defaults to 0.
        **kwargs : keyword arguments
            Additional attributes to set for the proposal.

        Returns
        -------
        bool
            `True` if the proposal was successfully sent and a response was received; otherwise, `False`.
        """
        if random:
            self.randomise_attributes()
        self._set_attributes(**kwargs)
        self.proposal_type = 'price_proposal'
        success = self._send()
        if not buy and success:
            display(self.last_msg)
            self.GUI_data.append(self.last_msg)
        return success
    
    def buy_contract(self, runs=1, random=False, **kwargs) -> int:
        """
        Attempts to buy a contract.

        This method tries to buy a contract for a specified number of `runs`. It randomizes 
        attributes if required and sends a price proposal with the `buy` parameter set to `True`. 
        Upon success, it updates the `proposal_id` and sends a buy contract proposal. The contract 
        ID is then appended to the `contract_ids` list.

        Parameters
        ----------
        runs : int, optional
            The number of contracts to attempt to buy. Defaults to 1.
        random : bool, optional
            Whether to randomize attributes before sending the proposal. Defaults to `False`.
        **kwargs : keyword arguments
            Additional attributes to set for the proposal.

        Returns
        -------
        int
            Returns 1 upon successful buy else 0.
        """

        i = 0
        temp_flag = 0
        while i < runs:
            i += 1
            if random or runs > 1:
                self.randomise_attributes()
            self._set_attributes(**kwargs)
            if self.price_proposal(buy=True):
                self.proposal_id = self.last_msg['proposal']['id']
                self.proposal_type = 'buy_contract'
                temp_flag = self._send()
                if temp_flag:
                    self.contract_ids.append(self.last_msg['buy']['contract_id'])
                    clear_output(wait=True)
                    print(i)
                    display(self.contract_proposal())
        return temp_flag
        
    def contract_proposal(self, contract_id=None, subscribe=0) -> None:
        """
        Gets the contract proposal.

        This method retrieves the details of a contract using the provided `contract_id`. 
        If no `contract_id` is provided, it uses the last saved contract ID. It sends a 
        contract proposal message and optionally subscribes to updates based on the `subscribe` parameter. 
        If `subscribe` is set to 1, it starts streaming the contract updates.

        Parameters
        ----------
        contract_id : str, optional
            The ID of the contract to retrieve details for. If not provided, the last saved contract ID is used.
        subscribe : int, optional
            Whether to subscribe to updates for the contract. Defaults to 0 (no subscription). If set to 1, 
            streaming of contract updates is initiated.

        Returns
        -------
        None
        """
        if contract_id == None:
            try:
                self.contract_id = self.contract_ids[-1]
            except Exception as e: 
                print("No Contract ID saved.")
        else:
            self.contract_id = contract_id
        self.proposal_type = 'contract_proposal'
        self.subscribe = subscribe
        if self.subscribe == 0:
            self._send()
        elif self.subscribe == 1:
            self.stream()
        self.GUI_data.append(self.last_msg)
        return self.last_msg
            
    def get_all_poc(self) -> None:
        """
        Retrieves all open contract proposals.

        This method iterates through all contract IDs stored in `self.contract_ids` and retrieves the details 
        for each open contract by calling the `contract_proposal` method with each contract ID.

        Returns
        -------
        None
        """
        for temp_id in self.contract_ids:
            clear_output(wait=True)
            self.contract_proposal(contract_id=temp_id)
            
    def tick_history(self, symbol=False, count=1, end='latest') -> dict:
        """
        Fetches historical tick data for the specified symbol.

        This method constructs a message to request historical tick data from the WebSocket API. It sends the 
        request with parameters specifying the symbol, number of ticks to retrieve, and the end time.

        Parameters
        ----------
        symbol : str, optional
            The symbol for which to retrieve tick history. If not provided, defaults to `self.symbol`.
        count : int, optional
            The number of ticks to retrieve. Defaults to 1.
        end : str, optional
            The end time for the tick data. Defaults to 'latest'.

        Returns
        -------
        dict
            The response from the WebSocket API containing the tick history data.
        """
        temp_msg = {
            "ticks_history": symbol if symbol else self.symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": end,
            "start": 1,
            "style": "ticks"
        }
        return self.send_message(temp_msg)
        
    def _check_connection(self) -> None:
        """
        Checks the WebSocket connection by sending a ping message and waiting for a response. If an exception 
        occurs (indicating the connection is closed), it attempts to reconnect by creating a new WebSocket 
        connection.

        Returns
        -------
        None
        """
        try:
            # Send a ping message
            self.ws.send(json.dumps({"ping": 1}))
            # Receive a response
            self.ws.recv()
        except Exception as e:
            print('Connection Closed, Reconnecting')
            time.sleep(0.2)
            clear_output(wait=True)
            self.create_ws()
            
    def _check_closed(self) -> int:
        """
        Checks if the contract has expired or been sold.

        This method inspects the latest message received from the WebSocket API to determine if the contract is 
        expired or sold.

        Returns
        -------
        int
            1 if the contract is expired or sold, otherwise 0.
        """
        if self.last_msg['proposal_open_contract']['is_expired'] or self.last_msg['proposal_open_contract']['is_sold']:
            return 1
        else: 
            return 0
            
    def sell_contract(self, contract_id=None) -> int:
        """
        Attempts to sell a contract by its ID.

        This method sets the proposal type to 'sell_contract' and sends the proposal to the WebSocket API. 
        It handles the response and displays the result. If the proposal is unsuccessful, it retrieves the 
        contract proposal details again.

        Parameters
        ----------
        contract_id : str, optional
            The ID of the contract to sell. If not provided, the last saved contract ID is used.

        Returns
        -------
        int
            Returns 1 upon successful sell else 0.
        """
        if contract_id == None:
            self.contract_id = self.contract_ids[-1]
        else:
            self.contract_id = contract_id
        self.proposal_type = 'sell_contract'
        temp_flag = self._send()
        if temp_flag:
            display(self.last_msg)
            # self.GUI_data.append(self.last_msg)
            self.contract_proposal(self.contract_id)
        else:
            self.contract_proposal(self.contract_id)
        return temp_flag

    def sell_all_contracts(self) -> None:
        """
        Sells all contracts in the contract IDs list.

        This method iterates over the list of contract IDs, and attempts to sell each contract one by one
        by calling the `sell_contract` method

        Returns
        -------
        None
        """ 
        for temp_id in self.contract_ids:
            clear_output(wait=True)
            self.sell_contract(contract_id=temp_id)
    
    def create_ws(self, **kwargs) -> None:
        """
        Creates a WebSocket connection to the specified endpoint.

        This method initializes the WebSocket connection using the endpoint URL. It sets the `ws` attribute 
        to the WebSocket connection object.

        Returns
        -------
        None
        """
        self.ws = 0
        self._set_attributes(**kwargs)
        self.ws = websocket.create_connection(self.endpoint)

    def authorise(self, **kwargs) -> int:
        """
        Authorizes the WebSocket connection using the provided API token.

        This method sends an authorization request with the API token and checks the response to determine 
        if the authorization was successful.

        Returns
        -------
        int
            1 if authorization is successful, otherwise 0.
        """
        self._check_connection()
        self._set_attributes(**kwargs)
        self.ws.send(json.dumps({"authorize": self.api_token}))
        if self._recv_msg_handler():
            return 1
        else:
            return 0
    
    def _recv_msg_handler(self) -> int:
        """
        Handles the reception of messages from the WebSocket.

        This method attempts to receive and process messages from the WebSocket. It handles errors and retries 
        sending the request if necessary. Specific errors are managed with different strategies, such as reauthorizing 
        or adjusting contract parameters based on the error details.

        Returns
        -------
        int
            1 if the message is received and processed successfully, otherwise 0.
        """

        try:
            temp = self.ws.recv()
            self.last_msg = json.loads(temp)
        except Exception as e:
            self.retry = 0
            return self._send()
        self.messages.append(self.last_msg)
        error_handle_flag = 0
        if 'error' in self.last_msg:
            self.retry += 1
            error = self.last_msg['error']
            match error['code']:
                case 'RateLimit':
                    self.create_ws()
                    self.retry +=1
                case 'InvalidToken' | 'OpenPositionLimitExceeded' | 'InvalidSellContractProposal' | 'InvalidtoSell' | 'InvalidOfferings':
                    display(error)
                    error_handle_flag = 1 
                case 'AuthorizationRequired':
                    if self.authorise():
                        self.retry +=1
                    else:
                        display('Auth failed', error)
                        error_handle_flag = 1 
                case 'OfferingsValidationError':
                    if 'Number of ticks must be between' in error['message']:
                        arr = list(map(int, re.findall(r'\d+', error['message'])))
                        self.duration = random.randint(arr[0], arr[1])
                    elif error['message'] == 'Trading is not offered for this asset.': 
                        self.symbol = 'R_100'        
                    else:
                        display(error)
                        error_handle_flag = 1 
                case 'ContractBuyValidationError' | 'OfferingsValidationFailure':
                    if 'message' in error.keys() and error['message'] == 'This trade is temporarily unavailable.':
                        display(error)
                        error_handle_flag = 1 
                    elif error['details']['field'] == 'barrier':
                        try: 
                            self.strike = error['details']['barrier_choices'][0]
                            # self.strike = random.choice(error['details']['barrier_choices'])
                        except Exception as e:
                            self.strike = '1000'
                    elif error['details']['field'] == 'amount':
                        # display(self.last_msg)
                        self.stake = error['details']['min_stake']
                    elif error['details']['field'] == 'duration':
                        self.duration = np.random.randint(1,200)
                    elif error['details']['field'] == 'multiplier':
                        self.multiplier = random.choice(list(map(int, re.findall(r'\d+', self.last_msg['error']['message']))))
                    elif error['details']['field'] == 'payout_per_point':
                        self.payout_per_point = random.choice(error['details']['payout_per_point_choices'])
                    else:
                        display(error)
                        error_handle_flag = 1 
                case 'InvalidContractProposal':
                    self.buy_contract()
                case 'ContractCreationFailure':
                    self.randomise_attributes()
                case _: # Default case
                    print('Undefined Error')
                    display(error)
                    error_handle_flag = 1 
            if error_handle_flag or self.retry > 8:
                self.retry = 0
                return 0                
            elif self.retry > 0: 
                return self._send()
            else:
                self.retry = 0
                return 0
        else:
            self.retry = 0
            return 1

def accu_barriers(prop) -> tuple:
    """
    Determines if the higher, lower, and barrier distance of the current spot are accurate for an accumulator contract.

    Parameters
    ----------
    prop : dict
        The proposal dictionary containing contract details and spot information.

    Returns
    -------
    tuple
        A tuple containing the calculated high barrier, low barrier, barrier distance, and a boolean indicating 
        if the barriers and distance are accurate.
    """
    contract_details = prop['proposal']['contract_details']
    spot = prop['proposal']['spot']
    symbol = prop['echo_req']['symbol']
    tick_size_barrier = symbols[symbol]['accu']['barrier'][str(prop['echo_req']['growth_rate'])]

    high_bar, low_bar, bar_dist = \
        (round(spot * (1+tick_size_barrier), symbols[symbol]['precision']+1)), \
        (round(spot * (1-tick_size_barrier), symbols[symbol]['precision']+1)), \
        (round(spot * (tick_size_barrier), symbols[symbol]['precision']+1))
    pass_fail =\
    abs(float(contract_details['high_barrier']) - high_bar) <= 1/10**symbols[symbol]['precision'] and \
    abs(float(contract_details['low_barrier']) - low_bar) <= 1/10**symbols[symbol]['precision'] and \
    abs(float(contract_details['barrier_spot_distance']) - bar_dist) <= 1/10**symbols[symbol]['precision']
    return high_bar, low_bar, bar_dist, pass_fail

def accu_payout(prop) -> tuple:
    """
    Calculates the payout, match flag, and close flag for an accumulator contract.

    Parameters
    ----------
    prop : dict
        The proposal open contract dictionary containing contract details, spot information, and other related data.

    Returns
    -------
    tuple
        A tuple containing the calculated payout, a boolean indicating if the payout matches the bid price, 
        and a string indicating the close flag status.
    """
    symbol = prop['proposal_open_contract']['underlying']
    stake = get_stake(prop)
    growth_rate = prop['proposal_open_contract']['growth_rate']
    barrier = symbols[symbol]['accu']['barrier'][str(growth_rate)]
    currency = prop['proposal_open_contract']['currency']
    ticks = prop['proposal_open_contract']['tick_passed']
    ko_flag = 0
    close_flag = 0
    if prop['proposal_open_contract']['is_sold']:
        if prop['proposal_open_contract']['profit'] < 0:
            end_spots = []
            for i in prop['proposal_open_contract']['audit_details']['all_ticks']:
                if 'flag' in i and 'Exit' in i['name']:
                    end_spots.append(i['tick'])
            ko_flag = 1 if abs(end_spots[1] - end_spots[0]) > end_spots[0] * barrier else 0
        close_flag = 'ko' if ko_flag else 'sold'
    else: 
        close_flag = 'bid'
    payout = np.round(stake * (1+growth_rate) ** ticks, BO['currency_precision'][currency])
    payout = 0 if ko_flag else payout
    match_flag = payout == prop['proposal_open_contract']['bid_price']
    return payout, match_flag, close_flag

class Accumulator(Product):
    """
    A class to represent an Accumulator product, inheriting from the Product class.

    Attributes:
        species (str): The type of the product, 'Accumulator'.
        contract_type (str): The type of contract, default is 'ACCU'.
        growth_rate (float): The growth rate of the accumulator, default is 0.05.

    Methods:
        randomise_attributes(): Randomises the attributes of the Accumulator instance.
        _proposal(): Generates a proposal specific to the Accumulator product.
        check_barrier(runs=0, show=False): Checks barrier conditions for the given number of runs.
        check_payout(runs=0, show=False): Checks payout conditions for the given number of runs.
    """
    
    species = "Accumulator"
    
    def __init__(self, **kwargs) -> None:
        """
        Initializes an Accumulator instance.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to initialize the instance attributes. Includes 'growth_rate', which defaults to 0.05.

        Returns
        -------
        None
        """        
        super().__init__(**kwargs)
        self.contract_type = 'ACCU'
        self.growth_rate = kwargs.get('growth_rate', 0.05)
        self.user_input_params = ['stake','symbol','growth_rate','take_profit']

    def randomise_attributes(self) -> None:
        """
        Randomises the attributes of the Accumulator instance.

        Sets the symbol to a random choice from available symbols, 
        the stake to a random value within the defined range, and
        the growth rate to a random value from a predefined list.

        Returns
        -------
        None
        """
        self.symbol = random.choice(list(symbols.keys()))
        self.stake = round(np.random.uniform(1,BO['accumulators']['max_stake'][self.currency]), BO['currency_precision'][self.currency])
        self.growth_rate = random.choice([0.01,0.02,0.03,0.04,0.05])

    def _proposal(self) -> str:
        """
        Generates a proposal specific to the Accumulator product.

        Creates a proposal dictionary based on the current instance attributes.
        Includes 'take_profit' and 'stop_loss' if they are set. 

        Returns
        -------
        str
            The JSON-encoded proposal.
        """
        proposal = {}
        if self.proposal_type == 'price_proposal':
            proposal = {
                "proposal": 1,
                "amount": self.stake,
                "basis": "stake",
                "contract_type": self.contract_type,
                "currency": self.currency,
                "symbol": self.symbol,
                'growth_rate' : self.growth_rate
            }
            if self.take_profit:
                insert_dict(proposal, 'limit_order', {'take_profit': self.take_profit})
            if self.stop_loss:
                insert_dict(proposal, 'limit_order', {'stop_loss': self.stop_loss})
        else:
            return super()._proposal()
        return json.dumps(proposal)
    
    def check_barrier(self, runs=0, show = False, **kwargs) -> None:
        """
        Checks barrier conditions for all saved proposals and the given number of runs.

        Randomises attributes and generates price proposals for the specified number of runs.
        Analyzes messages to check if barriers are accurate and displays the results.

        Parameters
        ----------
        runs : int
            The number of times to run the check.
        show : bool
            If True, displays messages with failed barrier checks.
        **kwargs : dict
            Additional keyword arguments for setting attributes.

        Returns
        -------
        None
        """
        for i in range(runs):
            clear_output(wait = True)
            self.randomise_attributes()
            self._set_attributes(**kwargs)
            self.price_proposal()
        clear_output(wait = True)
        self.stats['barrier_check'] = {'cases': 0, 'pass': 0, 'fail': 0}        
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] == 'proposal' and 'error' not in item and 'ACCU' in item['echo_req']['contract_type']:
                temp_dict = {}
                temp_dict['high_barrier'], temp_dict['low_barrier'], temp_dict['barrier_spot_distance'], temp_dict['barrier_check'] = accu_barriers(item)
                insert_dict(item, 'mv', temp_dict)
                self.stats['barrier_check']['cases'] += 1
                if item['mv']['barrier_check']: 
                    self.stats['barrier_check']['pass'] += 1
                else: 
                    self.stats['barrier_check']['fail'] += 1                
                if not item['mv']['barrier_check'] and show:
                    display(item)
        self.stats_barrier = dict_to_df(self.stats['barrier_check'], 'Payout')   
        return self.stats_barrier
        
    def check_payout(self, runs=0, show=False, **kwargs) -> pd.DataFrame:
        """
        Validates payout calculations for accumulator contracts over multiple runs.

        This method iterates through multiple payout checks by buying contracts, 
        retrieving all open proposals, and comparing the actual payout with 
        calculated values.

        Parameters
        ----------
        runs : int, optional
            Number of iterations to perform payout checks. Defaults to 0.
        show : bool, optional
            Whether to display proposals with failed payout checks. Defaults to False.
        **kwargs : dict, optional
            Additional parameters for buying contracts (e.g., attributes to set).

        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing the payout validation statistics.

        Notes
        -----
        - The method organizes the results into various outcome categories:
            - `total`: All cases processed.
            - `ko`: Knock-out contracts.
            - `sold`: Successfully sold contracts.
            - `bid`: Contracts still in bid state.
        - Failed payout checks are displayed if `show=True`.

        Examples
        --------
        >>> obj = Accumulator()
        >>> obj.check_payout(runs=10, show=True)
        Cases for Payout check:
        Category  cases  pass  fail
        0     total     10     9     1
        1        ko      2     2     0
        2      sold      6     5     1
        3       bid      2     2     0
        """
        for i in range(runs):
            clear_output(wait = True)
            self.buy_contract(random=1, **kwargs)
        self.get_all_poc()
        clear_output(wait = True)
        self.stats['payout_check'] = {
            'total': {'cases': 0, 'pass': 0, 'fail': 0},
            'ko': {'cases': 0, 'pass': 0, 'fail': 0},
            'sold': {'cases': 0, 'pass': 0, 'fail': 0},
            'bid': {'cases': 0, 'pass': 0, 'fail': 0}
        }
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] == 'proposal_open_contract' and 'error' not in item and 'ACCU' in item['proposal_open_contract']['contract_type']:
                temp_dict = {}
                temp_dict['payout'], temp_dict['payout_check'], temp_dict['close_flag'] = accu_payout(item) 
                insert_dict(item, 'mv', temp_dict)
                self.stats['payout_check'][temp_dict['close_flag']]['cases'] += 1
                self.stats['payout_check']['total']['cases'] += 1
                if item['mv']['payout_check']: 
                    self.stats['payout_check'][temp_dict['close_flag']]['pass'] += 1
                    self.stats['payout_check']['total']['pass'] += 1
                else: 
                    self.stats['payout_check'][temp_dict['close_flag']]['fail'] += 1  
                    self.stats['payout_check']['total']['fail'] += 1
                if not item['mv']['payout_check'] and show:
                    display(item) 
        self.stats_payout = dict_to_df(self.stats['payout_check'], 'Payout')
        return self.stats_payout

def vanilla_barriers(prop) -> list:
    """
    Calculate expected barrier choices for vanillas based on given JSON.

    Parameters
    ----------
    prop : dict
        The JSON proposal containing necessary data for calculations.

    Returns
    -------
    list
        A list of expected barrier choices.
    """
    spot = prop['proposal']['spot']
    symbol = prop['echo_req']['symbol']
    duration = [prop['proposal']['date_start'], prop['proposal']['date_expiry']]
    duration_unit = prop['echo_req']['duration_unit']
    digits = symbols[symbol]['digits']
    sigma = int(re.search(r'(\d+)[^\d]*$', symbol).group(1))/100
    n_digits = round(-np.log10(spot * sigma * np.sqrt(60 / 365 / 86400)), 0) - 1
    spot = round(spot / 10 ** -n_digits) * 10 ** -n_digits
    t = (int(duration[1]) - int(duration[0])) / 365 / 86400    
    if t < 5 * 60 / 365 / 86400 and 'R' in symbol:
        t -= 2 / 365 / 86400        
    if duration_unit != 'd':
        deltas = np.array(BO['vanillas']['delta_config'])
        real_strikes = spot * np.exp((sigma ** 2) / 2 * t - (sigma * np.sqrt(t) * norm.ppf(deltas)))
        displayed_barriers = np.round((real_strikes - spot) * 10 ** digits) / 10 ** digits
    else:
        temp_barriers = []
        displayed_barriers = []
        central_strike = round(spot / (10 ** -n_digits)) * (10 ** -n_digits)
        max_strike = np.floor((spot * np.exp((sigma ** 2) / 2 * t - (sigma * np.sqrt(t) * norm.ppf(0.1)))) / (10 ** -n_digits)) * (10 ** -n_digits)
        min_strike = np.ceil((spot * np.exp((sigma ** 2) / 2 * t - (sigma * np.sqrt(t) * norm.ppf(0.9)))) / (10 ** -n_digits)) * (10 ** -n_digits)
        strike_step_1 = round((central_strike - min_strike) / 5 / 10) * 10
        strike_step_2 = round((max_strike - central_strike) / 5 / 10) * 10         
        for i in range(5):
            temp_barriers.append(min_strike + strike_step_1 * i)            
        for i in range(5):
            temp_barriers.append(central_strike + strike_step_2 * i)            
        temp_barriers.append(max_strike)        
        for strike in temp_barriers:
            if strike not in displayed_barriers:
                displayed_barriers.append(strike)
    return displayed_barriers

def vanilla_ppp_ask(prop) -> float:
    """
    Calculates the payout per point (PPP) ask price for a vanilla contract.

    This function computes the expected PPP ask price for a vanilla contract based on the 
    Black-Scholes pricing model with adjustments for volatility charges, critical financial 
    variables, and the contract's type (call or put). It also validates the computed PPP 
    against the value provided in the proposal.

    Parameters
    ----------
    prop : dict
        The contract proposal data, which contains information about contract details.
        It can be in either of the following formats:
        - If `prop['msg_type'] == 'proposal_open_contract'`, the open contract details.
        - Otherwise, the initial contract proposal.

    Returns
    -------
    tuple
        A tuple containing:
        - ppp (float): The calculated payout per point ask price.
        - match_flag (bool): Whether the calculated price matches the provided API price.

    Notes
    -----
    - The method uses the Black-Scholes pricing model with adjustments for volatility 
      charges, spot spread, and markup values.
    - It handles both long call and long put vanilla contracts.
    - If the contract duration is short and involves synthetic indices ('R'), the 
      duration is reduced slightly to account for specific model adjustments.

    Examples
    --------
    >>> prop = {
    ...     'msg_type': 'proposal',
    ...     'proposal': {
    ...         'spot': 100,
    ...         'contract_details': {'barrier': "105"},
    ...         'display_number_of_contracts': "5",
    ...         'date_start': 1650000000,
    ...         'date_expiry': 1650003600
    ...     },
    ...     'echo_req': {
    ...         'symbol': 'R_100',
    ...         'contract_type': 'VANILLALONGCALL'
    ...     }
    ... }
    >>> ppp, match_flag = vanilla_ppp_ask(prop)
    >>> print(ppp, match_flag)
    1.234, True
    """    
    if prop['msg_type'] == 'proposal_open_contract':
        spot = prop['proposal_open_contract']['entry_spot']
        strike = float(prop['proposal_open_contract']['barrier'])      
        number_of_contracts = prop['proposal_open_contract']['display_number_of_contracts']
        api_ppp = float(prop['proposal_open_contract']['display_number_of_contracts'])
        symbol = prop['proposal_open_contract']['underlying']
        r = 0 if 'pricing_args' not in prop['proposal_open_contract'] else prop['proposal_open_contract']['pricing_args']['discount_rate']
        contract_type = prop['proposal_open_contract']['contract_type']
        duration = [prop['proposal_open_contract']['date_start'], prop['proposal_open_contract']['date_expiry']]
        stake = get_stake(prop)
        currency = prop['proposal_open_contract']['currency']
    else:
        spot = prop['proposal']['spot']
        strike = float(prop['proposal']['contract_details']['barrier'])      
        number_of_contracts = prop['proposal']['display_number_of_contracts']
        api_ppp = float(prop['proposal']['display_number_of_contracts'])
        symbol = prop['echo_req']['symbol']    
        r = 0 if 'pricing_args' not in prop['proposal'] else prop['proposal']['pricing_args']['discount_rate']
        contract_type = prop['echo_req']['contract_type']
        duration = [prop['proposal']['date_start'], prop['proposal']['date_expiry']]
        stake = get_stake(prop)
        currency = prop['echo_req']['currency']
    t = ( int(duration[1]) - int(duration[0]) ) / 365 / 86400  
    if t < 5 * 60 / 365 / 86400 and 'R' in symbol:
        t -= 2 / 365 / 86400
    vol_charge = BO['vanillas']['vol_markup']['daily'][symbol] if t > 1/365 else BO['vanillas']['vol_markup']['intra'][symbol]
    vol_charge = 1 + vol_charge
    sigma = get_sigma(symbol)
    spot_spread = BO['vanillas']['spread_spot'][symbol]
    bs_markup = BO['vanillas']['bs_markup']        
    vol = sigma * vol_charge
    d1 = (np.log(spot/strike) + (vol**2/2)*t)/(vol*np.sqrt(t))
    d2 = (d1 - vol*np.sqrt(t))
    if contract_type == "VANILLALONGCALL":
        phi = 1        
        delta_charge =  (norm.cdf((np.log(spot/strike) + (sigma**2/2)*t)/(sigma*np.sqrt(t)))) * spot_spread / 2        
    elif  contract_type == "VANILLALONGPUT":        
        phi = -1        
        delta_charge =  (abs(norm.cdf((np.log(spot/strike) + (sigma**2/2)*t)/(sigma*np.sqrt(t))) - 1)) * spot_spread / 2      
    bs_price = (phi*np.exp(-r*t)*(spot*norm.cdf(phi*d1) - strike*norm.cdf(phi*d2)))    
    ppp = rounddown(stake / ( bs_price + delta_charge + bs_markup ), BO['currency_precision'][currency]+4)
    return ppp, (ppp == api_ppp) 

def vanilla_payout(prop) -> tuple:
    """
    Calculate payout for a vanilla contract and determine if it matches the expected value.

    Parameters
    ----------
    prop : dict
        The JSON proposal containing necessary data for calculations.

    Returns
    -------
    tuple
        A tuple containing the calculated payout, a match flag indicating if the calculated
        payout matches the proposal value, and a close flag indicating the status of the contract.
    """
    symbol = prop['proposal_open_contract']['underlying']
    ctype = prop['proposal_open_contract']['contract_type']
    phi = 1 if ctype == "VANILLALONGCALL" else -1
    try:
        spot = prop['proposal_open_contract']['exit_tick'] 
    except Exception as e:
        spot = prop['proposal_open_contract']['current_spot']
    strike = float(prop['proposal_open_contract']['barrier'])
    ppp = float(prop['proposal_open_contract']['display_number_of_contracts'])
    try:
        tick_now = prop['proposal_open_contract']['exit_tick_time']
    except Exception as e:                
        tick_now = prop['proposal_open_contract']['current_spot_time']
    R_expire_flag = True if 'R' in symbol and prop['proposal_open_contract']['expiry_time']%2 and prop['proposal_open_contract']['expiry_time'] - tick_now == 1 else False
    if tick_now >= prop['proposal_open_contract']['expiry_time'] or R_expire_flag:
        payout, close_flag = np.round(max((phi * ppp * (spot - strike)), 0), BO['currency_precision'][prop['proposal_open_contract']['currency']]), 'expired'
        match_flag = payout == prop['proposal_open_contract']['sell_price']
    else:
        try:
            t = (prop['proposal_open_contract']['date_expiry'] - tick_now) / 365 / 86400
        except Exception as e:                
            t = (prop['proposal_open_contract']['date_expiry'] - tick_now) / 365 / 86400
        if t < 5 * 60 / 365 / 86400 and 'R' in symbol:
            t -= 2 / 365 / 86400   
        r = 0 if 'pricing_args' not in prop['proposal_open_contract'] else prop['proposal_open_contract']['pricing_args']['discount_rate']
        sigma = get_sigma(symbol)
        vol = sigma * (1-(BO['vanillas']['vol_markup']['daily'][symbol] if t > 1/365 else BO['vanillas']['vol_markup']['intra'][symbol]))
        spot_spread = BO['vanillas']['spread_spot'][symbol]
        bs_markup = BO['vanillas']['bs_markup']
        if phi == 1:
            delta_charge = (norm.cdf((np.log(spot/strike) + (sigma**2/2)*t)/(sigma*np.sqrt(t)))) * spot_spread / 2        
        else:
            delta_charge = (abs(norm.cdf((np.log(spot/strike) + (sigma**2/2)*t)/(sigma*np.sqrt(t))) - 1)) * spot_spread / 2   
        d1 = (np.log(spot/strike) + (vol**2/2)*t)/(vol*np.sqrt(t))
        d2 = (d1 - vol*np.sqrt(t))    
        payout = np.round(max(((phi*np.exp(-r*t)*(spot*norm.cdf(phi*d1) - strike*norm.cdf(phi*d2))) - delta_charge - bs_markup) * ppp, 0), BO['currency_precision'][prop['proposal_open_contract']['currency']])
        match_flag = (payout == prop['proposal_open_contract']['bid_price']) if t < 1/365 else True
        close_flag = 'sold' if prop['proposal_open_contract']['is_sold'] else 'bid'
    return payout, match_flag, close_flag

class Vanilla(Product):
    """
    A class to represent a Vanilla product, inheriting from the Product class.

    Attributes
    ----------
    contract_type : str
        The type of contract, either 'VANILLALONGCALL' or 'VANILLALONGPUT'.
    strike : str
        The strike price for the contract.
    duration : int
        The duration of the contract.
    duration_unit : str
        The unit of the duration ('m' for minutes, 'h' for hours, 'd' for days).

    Methods
    -------
    randomise_attributes():
        Randomises the attributes of the Vanilla instance.
    _proposal():
        Generates a proposal specific to the Vanilla product.
    check_barrier(runs=0, show=False):
        Validates the offered barriers in all proposals and for the given number of runs if defined.
    check_ask(runs=0, show=False):
        Validates the payout per point (ask price) in all proposals and for the given number of runs if defined.
    check_payout(runs=0, show=False):
        Validates the payout (expired/bid price) in all proposals and for the given number of runs if defined.
    """
        
    species = "Vanilla"
    
    def __init__(self, **kwargs) -> None:
        """
        Initializes a Vanilla instance with given keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for initializing the Vanilla instance, including:
            - strike (str): The strike price for the contract (default is '+0.00').
            - contract_type (str): The type of contract, either 'call' or 'put' (default is 'call').
            - duration (int): The duration of the contract (default is 1).
            - duration_unit (str): The unit of the duration ('m' for minutes, 'h' for hours, 'd' for days, default is 'm').
        """
        super().__init__(**kwargs)
        self.strike = kwargs.get('strike', '+0.00')
        self.contract_type = 'VANILLALONGCALL' if 'c' in kwargs.get('contract_type', 'call') else 'VANILLALONGPUT'
        self.duration = kwargs.get('duration', 1)
        self.duration_unit = kwargs.get('duration_unit', 'm')
        self.user_input_params = ['stake','symbol','duration_unit','duration']
            
    def randomise_attributes(self) -> None:
        """
        Randomize the attributes of the Vanilla instance.
        Randomizes the symbol, duration unit, strike price, contract type, and duration.
       
        Returns
        -------
        None
        """
        self.symbol = random.choice(list(symbols.keys()))
        self.duration_unit = random.choice(['m','h','d'])
        self.strike = '+0.00' if self.duration_unit != 'd' else '1000'
        self.contract_type = random.choice(['VANILLALONGCALL','VANILLALONGPUT'])
        self.duration = np.random.randint(1,61) if self.duration_unit == 'm' else\
                        np.random.randint(1,25) if self.duration_unit == 'h' else\
                        np.random.randint(1, (14 if self.symbol == '1HZ250V' else 40 if self.symbol == '1HZ150V' else 365)) 
        
    def _proposal(self) -> str:
        """
        Generate a proposal specific to the Vanilla product.

        Returns
        -------
        str
            The proposal in JSON format.
        """
        proposal = {}
        if self.proposal_type == 'price_proposal':
            proposal = {
                "proposal": 1,
                "amount": self.stake,
                "basis": "stake",
                "contract_type": self.contract_type,
                "currency": self.currency,
                "symbol": self.symbol,
                'barrier' : self.strike,
                'duration' : self.duration,
                'duration_unit' : self.duration_unit
            }
        else:
            return super()._proposal()
        return json.dumps(proposal)

    def check_barrier(self, runs=0, show = False, **kwargs) -> None:
        """
        Checks offered barrier choices for all saved proposals and the given number of runs.

        Randomises attributes and generates price proposals for the specified number of runs.
        Analyzes messages to check if barriers are accurate and displays the results.

        Parameters
        ----------
        runs : int
            The number of times to run the check.
        show : bool
            If True, displays messages with failed barrier checks.
        **kwargs : dict
            Additional keyword arguments for setting attributes.

        Returns
        -------
        None

        Notes
        -----
        Updates the `stats['barrier_stats']` attribute with the validation results.
        """
        for i in range(runs):
            clear_output(wait = True)
            self.randomise_attributes()
            self._set_attributes(**kwargs)
            self.price_proposal()
        clear_output(wait = True)
        self.stats['barrier_stats'] = {'cases': 0, 'pass': 0, 'fail': 0}
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] == 'proposal' and 'error' not in item and 'VANILLA' in item['echo_req']['contract_type']:
                temp_dict = {'barrier_choices': vanilla_barriers(item)}
                insert_dict(item, 'mv', temp_dict)    
                try:
                    barrier_pass_fail =  all([float(bar) for bar in item['proposal']['barrier_choices']] == item['mv']['barrier_choices'])
                except:
                    item['mv']['barrier_choices'] = sorted(item['mv']['barrier_choices'])
                    barrier_pass_fail = [float(bar) for bar in item['proposal']['barrier_choices']] == item['mv']['barrier_choices']
                item['mv']['barrier_check'] = barrier_pass_fail
                self.stats['barrier_stats']['cases'] += 1
                if barrier_pass_fail: 
                    self.stats['barrier_stats']['pass'] += 1
                else: 
                    self.stats['barrier_stats']['fail'] += 1               
                if not barrier_pass_fail and show:
                    display(item)
        self.stats_barrier = dict_to_df(self.stats['barrier_stats'], 'Barrier')
        return self.stats_barrier

    def check_ask(self, runs=0, show = False, **kwargs) -> None:
        """
        Validate the payout per point (ask price) in all proposals and for the given number of runs if specified.

        Parameters
        ----------
        runs : int, optional
            The number of additional random contracts to validate (default is 0).
        show : bool, optional
            Whether to display failed validations (default is False).
        **kwargs : dict
            Additional attributes to set for the Vanilla instance.
        
        Returns
        -------
        None

        Notes
        -----
        Updates the `stats['ask_stats']` attribute with the validation results.
        """
        for i in range(runs):
            clear_output(wait = True)
            self.randomise_attributes()
            self._set_attributes(**kwargs)
            self.price_proposal()
            # self.buy_contract()
        clear_output(wait = True)
        self.stats['ask_stats'] = {'cases': 0, 'pass': 0, 'fail': 0}        
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] in ['proposal'] and 'error' not in item and 'VANILLA' in item['echo_req']['contract_type']:
            # and item['proposal_open_contract']['date_start'] == item['proposal_open_contract']['current_spot_time']:
                temp_dict = {}
                temp_dict['ask_ppp'], temp_dict['ask_ppp_check'] = vanilla_ppp_ask(item)                
                insert_dict(item, 'mv', temp_dict)
                self.stats['ask_stats']['cases'] += 1
                if item['mv']['ask_ppp_check']: 
                    self.stats['ask_stats']['pass'] += 1
                else: 
                    self.stats['ask_stats']['fail'] += 1                    
                if not item['mv']['ask_ppp_check'] and show:
                    display(item)
        self.stats_ask = dict_to_df(self.stats['ask_stats'], 'Ask Price')
        return self.stats_ask

    def check_payout(self, runs=0, show = False, **kwargs) -> None:
        """
        Validates the payout (bid price) in all saved proposals and for additional runs if specified.

        This method evaluates the correctness of payouts by calculating the bid price 
        for each proposal using the financial model and comparing it with the actual 
        value provided by the contract. The results are categorized by contract states.

        Parameters
        ----------
        runs : int, optional
            The number of additional random contracts to test (default is 0).
        show : bool, optional
            Whether to display proposals with failed payout validations (default is False).
        **kwargs : dict, optional
            Additional attributes or parameters for contract creation during testing.

        Returns
        -------
        None

        Notes
        -----
        - Results are stored in the `self.stats['payout_check']` dictionary, which tracks 
        the total cases and their outcomes (`pass` or `fail`) for contracts in the 
        following states:
            - `total`: All contracts evaluated during the method execution.
            - `expired`: Contracts that have expired.
            - `sold`: Contracts that are sold.
            - `bid`: Contracts in "bid" (active) state.
        - Failed payout checks are displayed if `show=True`.

        Examples
        --------
        >>> vanilla_product = Vanilla()
        >>> vanilla_product.check_payout(runs=10, show=True)
        Cases for Payout check:
        Category  cases  pass  fail
        0     total     10     9     1
        1   expired      2     2     0
        2      sold      6     5     1
        3       bid      2     2     0
        """
        for i in range(runs):
            clear_output(wait = True)
            self.buy_contract(random=1, **kwargs)
        self.get_all_poc()
        clear_output(wait = True)
        self.stats['payout_check'] = {
            'total': {'cases': 0, 'pass': 0, 'fail': 0},
            'expired': {'cases': 0, 'pass': 0, 'fail': 0},
            'sold': {'cases': 0, 'pass': 0, 'fail': 0},
            'bid': {'cases': 0, 'pass': 0, 'fail': 0}
        }
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] == 'proposal_open_contract' and 'error' not in item and 'VANILLA' in item['proposal_open_contract']['contract_type']:
                temp_dict = {}
                temp_dict['payout'], temp_dict['payout_check'], temp_dict['close_flag'] = vanilla_payout(item) 
                insert_dict(item, 'mv', temp_dict)
                self.stats['payout_check'][temp_dict['close_flag']]['cases'] += 1
                self.stats['payout_check']['total']['cases'] += 1
                if item['mv']['payout_check']: 
                    self.stats['payout_check'][temp_dict['close_flag']]['pass'] += 1
                    self.stats['payout_check']['total']['pass'] += 1
                else: 
                    self.stats['payout_check'][temp_dict['close_flag']]['fail'] += 1  
                    self.stats['payout_check']['total']['fail'] += 1
                if not item['mv']['payout_check'] and show:
                    display(item) 
        self.stats_payout = dict_to_df(self.stats['payout_check'], 'Payout')
        return self.stats_payout

def turbo_ask (symbol, spot, barrier, duration_unit, phi) -> float:
    """
    Calculates the ask price for a turbo contract.

    Parameters
    ----------
    symbol : str
        The underlying symbol for contract.
    spot : float
        The current spot price of the instrument.
    barrier : float
        The barrier level for the contract.
    duration_unit : str
        The unit of duration, e.g., 't' for ticks, 'd' for days, etc.
    phi : int
        The direction of the contract, 1 for long, -1 for short.

    Returns
    -------
    float
        The calculated ask price for the turbo contract.
    """
    tick_comm = 'tick_comm_' + ('up' if phi == 1 else 'down')
    tick_comm += '_t' if  duration_unit == 't' else  '_d' if duration_unit == 'd' else '_i'
    sigma = get_sigma(symbol)
    is_r = 2 if re.search(re.compile(r'R'), symbol) else 1
    if phi == 1:
        commup = BO['turbos'][tick_comm][symbol] * sigma * np.sqrt(is_r/365/86400)
        return spot * (1 + commup) - barrier
    else:
        commdown = BO['turbos'][tick_comm][symbol] * sigma * np.sqrt(is_r/365/86400)
        return barrier - spot * (1 - commdown) 

def turbo_barrier(prop, start_of_min_spot) -> float:
    """
    Calculate the expected barrier from the proposal.

    Parameters
    ----------
    prop : dict
        The proposal data as a dictionary.
    start_of_min_spot : float
        The spot price at the start of the minute of the contract.

    Returns
    -------
    float
        The calculated barrier level.
    """
    phi = 1 if prop['echo_req']['contract_type'] == 'TURBOSLONG' else -1
    symbol = prop['echo_req']['symbol']
    spot = start_of_min_spot
    ppp = prop['echo_req']['payout_per_point']
    stake = get_stake(prop)
    tick_comm = 'tick_comm_' + ('up' if phi == 1 else 'down')
    tick_comm += '_t' if  prop['echo_req']['duration_unit'] == 't' else  '_d' if prop['echo_req']['duration_unit'] == 'd' else '_i'
    sigma = get_sigma(symbol)
    is_r = 2 if re.search(re.compile(r'R'), symbol) else 1
    comm = BO['turbos'][tick_comm][symbol] * sigma * np.sqrt(is_r/365/86400)
    if phi == 1:
        barrier = roundup(spot * comm - stake/ppp, symbols[symbol]['precision'])
    else:
        barrier = rounddown(stake/ppp - spot * comm, symbols[symbol]['precision'])
    return round(prop['proposal']['spot'] + barrier, symbols[symbol]['precision']) 
    
def turbo_ppps(prop, start_of_min_spot) -> tuple[list, bool]:
    """
    Calculates the expected offered payout_per_point choices from the given json.

    Parameters
    ----------
    prop : dict
        The proposal data as a dictionary.
    start_of_min_spot : float
        The spot price at the start of the minute of the contract.

    Returns
    -------
    tuple
        A tuple containing the list of offered payout per point choices,
        a boolean indicating if the offered choices match the expected choices,
        and if the barrier matches the expected barrier.
    """
    stake = get_stake(prop)
    phi = 1 if 'LONG' in prop['echo_req']['contract_type'] else -1
    symbol = prop['echo_req']['symbol']
    spot = start_of_min_spot
    duration_unit = prop['echo_req']['duration_unit']
    sigma = int(re.search(r'(\d+)[^\d]*$', symbol).group(1))/100
    barrier_far = spot * (1 - phi * sigma * np.sqrt(BO['turbos']['T_max'][symbol]/365/86400))
    barrier_close = spot * (1 - phi * sigma * np.sqrt(BO['turbos']['T_min'][symbol]/365/86400))
    n_far = stake / turbo_ask(symbol, spot, barrier_far, duration_unit, phi)
    n_close = stake / turbo_ask(symbol, spot, barrier_close, duration_unit, phi)
    increment =  roundsf((n_close - n_far) * BO['turbos']['ip'][symbol])
    tn = roundup(n_far/increment,0) #tn is the smallests ppp
    t0 = rounddown(n_close/increment,0) #t0 is the largest
    ppp = [tn*increment]
    while ppp[-1] < t0*increment:
        ppp.append(round(ppp[-1]+increment,12))
    ppp = np.array(ppp)
    offered_pps = ppp[ppp <= round(t0*increment,12)]
    barrier = turbo_barrier(prop, spot)
    print(sorted(offered_pps) == sorted([float(x) for x in prop['proposal']['payout_choices']]))
    return offered_pps, barrier, sorted(offered_pps) == sorted([float(x) for x in prop['proposal']['payout_choices']]) and barrier == float(prop['proposal']['contract_details']['barrier'])

def turbo_payout (prop) -> tuple:
    """
    Calculates the payout for a turbo contract.

    Parameters
    ----------
    prop : dict
        The proposal data as a dictionary.

    Returns
    -------
    tuple
        A tuple containing the payout, a boolean indicating if the offered payout 
        matches the expected payout, and the close flag.
    """
    symbol = prop['proposal_open_contract']['underlying']
    is_r = 2 if re.search(re.compile(r'R'), symbol) else 1
    ctype = prop['proposal_open_contract']['contract_type']
    try:
        spot = prop['proposal_open_contract']['exit_tick'] 
    except Exception as e:
        spot = prop['proposal_open_contract']['current_spot']
    barrier = float(prop['proposal_open_contract']['barrier'])
    bid_price = prop['proposal_open_contract']['bid_price']
    ppp = float(prop['proposal_open_contract']['display_number_of_contracts'])
    shortcode = prop['proposal_open_contract']['shortcode']
    sigma = get_sigma(symbol)
    if 'exit_tick_time' in prop['proposal_open_contract'] and prop['proposal_open_contract']['exit_tick_time'] >= prop['proposal_open_contract']['expiry_time']:
        if (ctype == 'TURBOSLONG' and spot <= barrier) or (ctype == 'TURBOSSHORT' and spot >= barrier):
            payout, close_flag = 0, 'ko'
        else:
            payout, close_flag = abs(spot - barrier) * ppp, 'expired'
    else:
        comm = 'down_'if ctype == 'TURBOSLONG' else 'up_'
        comm += 't' if 'T' in shortcode[11:] else 'i' if prop['proposal_open_contract']['is_intraday'] else 'd'
        comm = 'tick_comm_' + comm
        comm = BO['turbos'][comm][symbol] * sigma * np.sqrt(is_r/365/86400)
        # print(comm)
        payout, close_flag = max( (spot*(1 - comm) - barrier) * ppp if ctype == 'TURBOSLONG' else (barrier - spot*(1 + comm)) * ppp, 0), 'sold' if prop['proposal_open_contract']['is_sold'] else 'bid'
    match_flag = np.round(payout,BO['currency_precision'][prop['proposal_open_contract']['currency']]) == bid_price
    return payout, match_flag, close_flag

class Turbo(Product):
    """
    A class to represent a Turbo product, inheriting from the Product class.

    Attributes
    ----------
    species : str
        The type of the product, 'turbo'.
    strike : str
        The strike price for the contract, default is '+10.00'.
    payout_per_point : float
        The payout per point, default is 2.
    contract_type : str
        The type of contract, either 'TURBOSLONG' or 'TURBOSSHORT'.
    duration : int
        The duration of the contract, default is 5.
    duration_unit : str
        The unit of the duration ('t' for ticks, 's' for seconds, 'm' for minutes, 'h' for hours, 'd' for days).

    Methods
    -------
    randomise_attributes()
        Randomises the attributes of the Turbo instance.
    _proposal()
        Generates a proposal specific to the Turbo product.
    get_start_of_min_tick(prop)
        Gets the start of the minimum tick price.
    check_ppps(runs=0, show=False, **kwargs)
        Checks the payout per point choices.
    check_payout(runs=0, show=False, **kwargs)
        Checks the payout for the Turbo product.
    """
    species = "Turbo"
    
    def __init__(self, **kwargs) -> None:
        """
        Initializes a new instance of the Turbo class.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for setting attributes.
            - strike (str): The strike price for the contract, default is '+10.00'.
            - payout_per_point (float): The payout per point, default is 2.
            - contract_type (str): The type of contract, either 'TURBOSLONG' or 'TURBOSSHORT'. Defaults to 'TURBOSLONG' if 'long' is in the contract type, otherwise 'TURBOSSHORT'.
            - duration (int): The duration of the contract, default is 5.
            - duration_unit (str): The unit of the duration, default is 'm'.
            - ver (str): The version of the Turbo, default is 'revamp'.
        """
        super().__init__(**kwargs)
        self.strike = kwargs.get('strike', '+10.00')
        self.payout_per_point = kwargs.get('payout_per_point', 2)
        self.contract_type = 'TURBOSLONG' if 'long' in kwargs.get('contract_type', 'TURBOSLONG') else 'TURBOSSHORT'
        self.duration = kwargs.get('duration', 5)
        self.duration_unit = kwargs.get('duration_unit', 'm')
        self.version = kwargs.get('ver', 'revamp')
        self.user_input_params = ['stake','symbol','duration_unit','duration']
            
    def randomise_attributes(self) -> None:
        """
        Randomises the attributes of the Turbo instance.
        
        The method randomly selects values for the symbol, duration unit, strike price,
        payout per point, contract type, and duration based on predefined ranges.
        """
        self.symbol = random.choice(list(symbols.keys()))
        self.stake = np.round(random.uniform(0.1, 2000), BO['currency_precision'][self.currency])
        self.duration_unit = random.choice(['t','s','m','h','d'])
        self.strike = '+10.00'
        self.payout_per_point = 2.00001
        self.contract_type = random.choice(['TURBOSLONG','TURBOSSHORT'])
        self.duration = np.random.randint(5, 11) if self.duration_unit == 't' else\
                        np.random.randint(15, 86401) if self.duration_unit == 's' else\
                        np.random.randint(1, 1441) if self.duration_unit == 'm' else\
                        np.random.randint(1, 25) if self.duration_unit == 'h' else\
                        np.random.randint(1, 365) 
        
    def _proposal(self) -> str:
        """
        Generates a proposal specific to the Turbo product.

        Returns
        -------
        str
            A JSON string representing the proposal.

        Notes
        -----
        If the proposal type is 'price_proposal', the method constructs a proposal 
        dictionary with the following keys:
        - proposal: int, set to 1.
        - amount: float, the stake amount.
        - basis: str, set to 'stake'.
        - contract_type: str, the contract type, either 'TURBOSLONG' or 'TURBOSSHORT'.
        - currency: str, the currency of the contract.
        - symbol: str, the underlying symbol symbol.
        - duration: int, the contract duration.
        - duration_unit: str, the unit of the duration.

        Depending on the version attribute:
        - If 'revamp', adds 'payout_per_point' to the proposal.
        - Otherwise, adds 'barrier' to the proposal.

        If the proposal type is not 'price_proposal', it calls the parent class's `_proposal` method.
        """        
        proposal = {}
        if self.proposal_type == 'price_proposal':
            proposal = {
                "proposal": 1,
                "amount": self.stake,
                "basis": "stake",
                "contract_type": self.contract_type,
                "currency": self.currency,
                "symbol": self.symbol,
                'duration' : self.duration,
                'duration_unit' : self.duration_unit
            }
            if self.version == 'revamp':
                proposal['payout_per_point'] = self.payout_per_point
            else:
                proposal['barrier'] = self.strike
            if self.take_profit:
                insert_dict(proposal, 'limit_order', {'take_profit': self.take_profit})
            if self.stop_loss:
                insert_dict(proposal, 'limit_order', {'stop_loss': self.stop_loss})
        else:
            return super()._proposal()
        return json.dumps(proposal)

    def get_start_of_min_tick(self, prop) -> float:
        """
        Retrieves the spot price at the start of the the minute 
        based on the entry tick for the given proposal.

        Parameters
        ----------
        prop : dict
            A dictionary representing the proposal. It must contain the following keys:
            - 'proposal': dict, containing 'date_start'.
            - 'echo_req': dict, containing 'symbol'.

        Returns
        -------
        float
            The spot price at the start of the the minute.

        Notes
        -----
        The method calculates the start of the minute based on the 'date_start' timestamp
        from the proposal. It then retrieves the tick history up to the start of the minute
        and returns the last price in the history.

        Examples
        --------
        >>> turbo = Turbo()
        >>> prop = {
        ...     'proposal': {'date_start': 1625140800},
        ...     'echo_req': {'symbol': 'R_100'}
        ... }
        >>> turbo.get_start_of_min_tick(prop)
        123.45
        """
        cur_time = prop['proposal']['date_start']
        start_of_min = cur_time - cur_time%60
        # time.sleep(0.5)
        return self.tick_history(symbol=prop['echo_req']['symbol'], end=start_of_min)['history']['prices'][-1]
            
    def check_ppps(self, runs=0, show = False, **kwargs) -> None:
        """
        Checks the offered payout per point (PPP) choices for all proposals and additional runs for the Turbo product.

        Parameters
        ----------
        runs : int, optional
            The number of runs to perform for checking PPP choices (default is 0).
        show : bool, optional
            Whether to display the proposals with failed PPP checks (default is False).
        **kwargs : dict, optional
            Additional keyword arguments to set attributes for the Turbo product.

        Notes
        -----
        This method performs the following steps:
        1. Randomizes the attributes and sets them.
        2. Generates price proposals and collects messages.
        3. Checks the PPP choices for each proposal.
        4. Updates the statistics for PPP choices.

        If `show` is True, it displays the proposals with failed PPP checks.

        Examples
        --------
        >>> turbo = mv.Turbo()
        >>> turbo.check_ppps(runs=5, show=True)
        """
        for i in range(runs):
            # time.sleep(0.71)
            clear_output(wait = True)
            self.randomise_attributes()
            self._set_attributes(**kwargs)
            self.price_proposal()
        clear_output(wait = True)
        self.stats['ppp_choices_stats'] = {'cases': 0, 'pass': 0, 'fail': 0}
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] == 'proposal' and 'error' not in item and 'TURBO' in item['echo_req']['contract_type']:
                # if 'mv' not in item or ('mv' in item and 'ppp_choices_stats' not in item['mv']): # some fixing needs to be done here to prevent lost of results.
                display(item)
                temp_dict = {}
                start_of_min_spot = self.get_start_of_min_tick(item)
                temp_dict['payout_choices'], temp_dict['barrier'], temp_dict['ppp_choices_stats'] = turbo_ppps(item, start_of_min_spot)
                insert_dict(item, 'mv', temp_dict)
                self.stats['ppp_choices_stats']['cases'] += 1
                if item['mv']['ppp_choices_stats']: 
                    self.stats['ppp_choices_stats']['pass'] += 1
                else: 
                    self.stats['ppp_choices_stats']['fail'] += 1
                clear_output(wait=True)
        if show:
            for item in self.messages:
                if 'msg_type' in item and item['msg_type'] == 'proposal' and 'error' not in item and not item['mv']['ppp_choices_stats']:
                    display(item)                     
        self.stats_ppp_choices = dict_to_df(self.stats['ppp_choices_stats'], 'PPP Choices')
        return self.stats_ppp_choices

    def check_payout(self, runs=0, show = False, **kwargs):
        """
        Checks the payout for all POCs and additional runs for the Turbo product.

        Parameters
        ----------
        runs : int, optional
            The number of runs to perform for checking payouts (default is 0).
        show : bool, optional
            Whether to display the POCs with failed payouts (default is False).
        **kwargs : dict, optional
            Additional keyword arguments to buy contracts.

        Notes
        -----
        This method performs the following steps:
        1. Buys contracts and collects messages.
        2. Retrieves all proposal open contracts (POCs).
        3. Checks the payout for each POC.
        4. Updates the statistics for payout checks.

        If `show` is True, it displays the proposals with failed payouts.

        Examples
        --------
        >>> turbo = mv.Turbo()
        >>> turbo.check_payout(runs=5, show=True)
        """
        for i in range(runs):
            clear_output(wait = True)
            self.buy_contract(random=1, **kwargs)
        self.get_all_poc()
        clear_output(wait = True)
        self.stats['payout_check'] = {
            'total': {'cases': 0, 'pass': 0, 'fail': 0},
            'ko': {'cases': 0, 'pass': 0, 'fail': 0},
            'expired': {'cases': 0, 'pass': 0, 'fail': 0},
            'sold': {'cases': 0, 'pass': 0, 'fail': 0},
            'bid': {'cases': 0, 'pass': 0, 'fail': 0}
        }
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] == 'proposal_open_contract' and 'error' not in item and 'TURBO' in item['proposal_open_contract']['contract_type']:
                temp_dict = {}
                temp_dict['payout'], temp_dict['payout_check'], temp_dict['close_flag'] = turbo_payout(item) 
                insert_dict(item, 'mv', temp_dict)
                self.stats['payout_check'][temp_dict['close_flag']]['cases'] += 1
                self.stats['payout_check']['total']['cases'] += 1
                if item['mv']['payout_check']: 
                    self.stats['payout_check'][temp_dict['close_flag']]['pass'] += 1
                    self.stats['payout_check']['total']['pass'] += 1
                else: 
                    self.stats['payout_check'][temp_dict['close_flag']]['fail'] += 1  
                    self.stats['payout_check']['total']['fail'] += 1
                if not item['mv']['payout_check'] and show:
                    display(item) 
        self.stats_payout = dict_to_df(self.stats['payout_check'], 'Payout')
        return self.stats_payout

def risefall_bid(prop) -> float:
    """
    Calculates the bid price for a Rise/Fall contract.

    This function computes the expected bid price for a Rise/Fall contract 
    using step-index logic for binary outcomes or the Black-Scholes pricing model for continuous outcomes.
    It compares the calculated bid price with the API-provided value and categorizes the contract's status.

    Parameters
    ----------
    prop : dict
        The contract data containing information such as the barrier, underlying spot price, contract type,
        duration, and payout.

    Returns
    -------
    tuple
        A tuple containing:
        - bid (float): The computed bid price.
        - match_flag (bool): True if the computed bid matches the API-provided bid price, otherwise False.
        - close_flag (str): Indicates the closure state of the contract ('expired', 'sold', or 'bid').

    Notes
    -----
    - For step-index contracts (`stp` in the symbol), a probabilistic calculation 
      involving Beta distribution is applied.
    - For non-step contracts, the bid price is calculated using a modified 
      Black-Scholes formula adjusted for commission.
    - Bid adjustments are applied based on contract status (e.g., `expired`).

    Examples
    --------
    >>> prop = {
    ...     'proposal_open_contract': {
    ...         'current_spot_time': 1657000000,
    ...         'date_expiry': 1657000100,
    ...         'barrier': 105.0,
    ...         'underlying': 'R_100',
    ...         'contract_type': 'CALL',
    ...         'bid_price': 12.5,
    ...         'payout': 100,
    ...         'current_spot': 104.0
    ...     }
    ... }
    >>> bid, match_flag, close_flag = risefall_bid(prop)
    >>> print(bid, match_flag, close_flag)
    12.34 True 'bid'
    """    
    duration = [prop['proposal_open_contract']['current_spot_time'], prop['proposal_open_contract']['date_expiry']]
    n = int(duration[1]) - int(duration[0])
    if 'tick' in prop['proposal_open_contract']['longcode']:
        n = list(map(int, re.findall(r'\d+', prop['proposal_open_contract']['longcode'])))[-1] - (prop['proposal_open_contract']['current_spot_time'] - prop['proposal_open_contract']['entry_tick_time'])
    t = n/365/86400   
    allow_equal = 1 if 'E' in prop['proposal_open_contract']['contract_type'] else 0
    
    # here is adjustment for be sus bid
    n-=1
    allow_equal = 0 if allow_equal else 1

    symbol = prop['proposal_open_contract']['underlying']
    contract_type = prop['proposal_open_contract']['contract_type']
    phi = 1 if 'C' in contract_type else -1
    is_step = 1 if 'stp' in symbol else 0 
    is_sold = prop['proposal_open_contract']['is_sold']
    if 'exit_tick' in prop['proposal_open_contract'] and prop['proposal_open_contract']['exit_tick_time'] >= prop['proposal_open_contract']['date_expiry']:
        expired = 1 
    else: expired = 0
    try:
        spot = float(prop['proposal_open_contract']['exit_tick'])     
    except Exception as e:
        spot = float(prop['proposal_open_contract']['current_spot'])      
    barrier = float(prop['proposal_open_contract']['barrier'])
    stake = get_stake(prop)
    bid_api = prop['proposal_open_contract']['bid_price']
    payout_api = prop['proposal_open_contract']['payout']
    commission = BO['risefall']['commission'][symbol]['tick' if 'T' in prop['proposal_open_contract']['shortcode'][4:] else 'second']
    commission /= 100
    if is_step:
        phi = 1 if "CALL" in contract_type else -1
        strike_dist = (barrier-spot)/symbols[symbol]['step_size']
        strike_dist = round(strike_dist)
        print(strike_dist, barrier, spot, symbols[symbol]['step_size'])
        # print(strike_dist)
        k = 0 
        p = 0.5
        print(n)
        if (n+strike_dist) % 2: #odd
            k = (n+strike_dist+1)/2 # k greater
            greater = betainc(k, n - k + 1, p)
            greater_equal = greater
            equal = 0 
        else: #even
            k = (n+strike_dist)/2
            greater = betainc(k+1, n - (k+1) + 1, p)
            greater_equal = betainc(k, n - (k) + 1, p)
            equal = greater_equal - greater
        smaller = 1 - greater_equal
        if allow_equal: 
            unit_price_bid = round(greater_equal,4) if phi==1 else round(smaller + equal,4)
        else:
            unit_price_bid = round(greater,4) if phi==1 else round(smaller,4)
        unit_price_bid -= commission
    else:
        sigma = get_sigma(symbol)
        d1 = (np.log(spot/barrier) + (sigma**2/2)*t)/(sigma*np.sqrt(t))
        d2 = (d1 - sigma*np.sqrt(t))
        unit_price_bid = norm.cdf(phi *d2) - commission
    bid = payout_api * unit_price_bid
    if expired: 
        match contract_type:
            case 'CALL':
                bid = payout_api if spot > barrier else 0
            case 'CALLE':
                bid = payout_api if spot >= barrier else 0
            case 'PUT':
                bid = payout_api if spot < barrier else 0
            case 'PUTE':
                bid = payout_api if spot <= barrier else 0            
    # print(unit_price_bid)
    # print(norm.cdf(phi *d2),unit_price_bid, phi, contract_type, commission)
    close_flag = 'expired' if expired else 'sold' if is_sold else 'bid'
    match_flag = np.round(bid, get_currency_precision(prop) ) == bid_api                
    return bid, match_flag, close_flag

def risefall_payout (prop) -> tuple:
    """
    Calculates the payout for a Rise/Fall contract and validates it.

    This function computes the expected payout for a Rise/Fall contract based on 
    its attributes, such as the barrier, entry spot, and duration. It determines 
    whether the computed payout matches the API-provided value and identifies the 
    contract's status.

    Parameters
    ----------
    prop : dict
        The Rise/Fall contract data containing details such as:
        - Barrier and spot prices.
        - Contract type (e.g., "CALL", "PUT").
        - Duration and time-related attributes.
        - Commission and payout information.

    Returns
    -------
    tuple
        A tuple containing:
        - payout (float): The computed payout amount.
        - match_flag (bool): True if the computed payout matches the API-provided 
          payout, otherwise False.
        - close_flag (str or bool): Indicates the status of the contract as `sold` 
          or a payout comparison result.

    Notes
    -----
    - For step-index contracts, a probabilistic calculation based on Beta distribution 
      is used to compute the payout.
    - For non-step contracts, the payout is calculated using a modified Black-Scholes 
      pricing formula with adjustments for commission.
    - If the computed payout is less than the API-provided payout, a `True` flag is returned 
      in place of the `close_flag`.

    Examples
    --------
    >>> prop = {
    ...     'proposal_open_contract': {
    ...         'entry_tick_time': 1657000000,
    ...         'date_expiry': 1657000100,
    ...         'barrier': 105.0,
    ...         'underlying': 'R_100',
    ...         'contract_type': 'CALL',
    ...         'shortcode': 'FR_tick',
    ...         'entry_spot': 104.0,
    ...         'payout': 100.0
    ...     }
    ... }
    >>> payout, match_flag, close_flag = risefall_payout(prop)
    >>> print(payout, match_flag, close_flag)
    95.2 True 'sold'
    """
    duration = [prop['proposal_open_contract']['entry_tick_time'], prop['proposal_open_contract']['date_expiry']]
    n = int(duration[1]) - int(duration[0])# - 1
    # print('t')
    n = list(map(int, re.findall(r'\d+', prop['proposal_open_contract']['longcode'])))[-1] if 'tick' in prop['proposal_open_contract']['longcode'] else n
    t = n/365/86400   
    allow_equal = 1 if 'E' in prop['proposal_open_contract']['contract_type'] else 0
    symbol = prop['proposal_open_contract']['underlying']
    contract_type = prop['proposal_open_contract']['contract_type']
    phi = 1 if 'C' in contract_type else -1
    is_step = 1 if 'stp' in symbol else 0 
    try:
        spot = float(prop['proposal_open_contract']['entry_spot'])
    except Exception as e:
        spot = float(prop['proposal_open_contract']['current_spot'])      
    try:
        barrier = float(prop['proposal_open_contract']['barrier'])
    except Exception as e:
        barrier = spot
    stake = get_stake(prop)
    payout_api = prop['proposal_open_contract']['payout']
    commission = BO['risefall']['commission'][symbol]['tick' if 'T' in prop['proposal_open_contract']['shortcode'][4:] else 'second']
    commission /= 100
    if is_step:
        phi = 1 if "CALL" in contract_type else -1
        strike_dist = round((barrier-spot)/symbols[symbol]['step_size'])
        print(barrier, spot, symbols[symbol]['step_size'], strike_dist,n)
        # print(strike_dist)
        k = 0 
        p = 0.5
        if (n+strike_dist) % 2: #odd
            k = (n+strike_dist+1)/2 # k greater
            # print(k)
            greater = betainc(k, n - k + 1, p)
            greater_equal = greater
            equal = 0 
        else: #even
            k = (n+strike_dist)/2
            print(k)
            greater = betainc(k+1, n - (k+1) + 1, p)
            greater_equal = betainc(k, n - (k) + 1, p)
            equal = greater_equal - greater
        smaller = 1 - greater_equal
        if allow_equal: 
            unit_price_ask = round(greater_equal,4) if phi==1 else round(smaller + equal,4)
        else:
            unit_price_ask = round(greater,4) if phi==1 else round(smaller,4)
        unit_price_ask += commission
    else:
        sigma = get_sigma(symbol)
        d1 = (np.log(spot/barrier) + (sigma**2/2)*t)/(sigma*np.sqrt(t))
        d2 = (d1 - sigma*np.sqrt(t))
        unit_price_ask = norm.cdf(phi *d2) + commission
    payout = stake/unit_price_ask
    # print(unit_price_ask)
    # print(allow_equal,unit_price_ask, phi, contract_type, commission)
    match_flag = np.round(payout, BO['currency_precision'][prop['proposal_open_contract']['currency']] ) == payout_api
    close_flag = 'sold'
    if np.round(payout, BO['currency_precision'][prop['proposal_open_contract']['currency']] ) < payout_api:
        close_flag = True
    return payout, match_flag, close_flag

class RiseFall(Product):
    """
    A class to represent a Rise/Fall contract, inheriting from the Product class.

    This class handles Rise/Fall contract-specific attributes, generating proposals, 
    and validating metrics such as bid prices and payout accuracy.

    Attributes
    ----------
    species : str
        The type of the product, always 'RiseFall'.
    contract_type : str
        The type of Rise/Fall contract (e.g., 'CALL', 'PUT'), default is 'CALL'.
    stake_type : str
        Determines whether calculation basis is 'stake' or 'payout'. Default is 'stake'.
    payout : float
        Default payout value if calculations depend on it.
    duration : int
        Duration of the contract in time units (default is 5).
    duration_unit : str
        The time unit for the duration (e.g., 't' for ticks, 's' for seconds, etc.). Default is 'm'.
    user_input_params : list
        List of customizable parameters for the Rise/Fall contract.
    is_step : bool
        Specifies whether the symbol is 'step-index'. Default is False.

    Methods
    -------
    randomise_attributes()
        Randomizes key contract attributes such as duration, symbol, stake, etc.
    _proposal()
        Generates a JSON-encoded proposal based on the instance's attributes.
    check_bid(runs=0, show=False, **kwargs)
        Validates bid price accuracy by calculating expected values for all proposals.
    check_payout(runs=0, show=False, **kwargs)
        Validates payout correctness for all previously saved proposals.
    """
    
    species = "RiseFall"
    
    def __init__(self, **kwargs) -> None:
        """
        Initializes a Rise/Fall instance.

        This constructor allows custom attributes for Rise/Fall-specific contracts 
        and sets default values if attributes are not specified.

        Parameters
        ----------
        **kwargs : dict
            Customizable attributes such as `stake_type`, `contract_type`, `payout`, `duration`, etc.

        Returns
        -------
        None
        """
        super().__init__(**kwargs)
        self.contract_type = kwargs.get('contract_type', 'CALL')
        self.stake_type = kwargs.get('stake_type', 'stake')
        self.payout = kwargs.get('payout', 10)
        self.duration = kwargs.get('duration', 5)
        self.duration_unit = kwargs.get('duration_unit', 'm')
        self.user_input_params = ['stake', 'symbol', 'duration', 'duration_unit', 'contract_type', 'stake_type']
        self.is_step = kwargs.get('is_step', False)

    def randomise_attributes(self) -> None:
        """
        Randomizes the attributes of the Rise/Fall contract instance.

        Adjusts attributes such as duration, contract type, stake, and symbol based on preset 
        conditions and ranges.

        Returns
        -------
        None
        """
        self.symbol = random.choice(list(symbols.keys()))
        if self.is_step:
            self.symbol = random.choice(['stpRNG', 'stpRNG2', 'stpRNG3', 'stpRNG4', 'stpRNG5'])
        self.duration_unit = random.choice(['t', 's', 'm', 'h', 'd'])
        self.contract_type = random.choice(['CALL', 'CALLE', 'PUT', 'PUTE'])
        self.duration = np.random.randint(1, 11) if self.duration_unit == 't' else \
                        np.random.randint(15, 86401) if self.duration_unit == 's' else \
                        np.random.randint(1, 1441) if self.duration_unit == 'm' else \
                        np.random.randint(1, 25) if self.duration_unit == 'h' else \
                        np.random.randint(1, 366)
        self.stake = round(np.random.uniform(1, 2000), BO['currency_precision'][self.currency])

    def _proposal(self) -> str:
        """
        Generates a JSON-encoded proposal for the Rise/Fall contract.

        Creates a proposal dictionary using instance attributes. Adds `take_profit` 
        and `stop_loss` limits if set.

        Returns
        -------
        str
            A JSON-encoded string representing a proposal.
        """
        proposal = {}
        if self.proposal_type == 'price_proposal':
            proposal = {
                "proposal": 1,
                "amount": self.stake if self.stake_type == 'stake' else self.payout,
                "basis": self.stake_type,
                "contract_type": self.contract_type,
                "currency": self.currency,
                "symbol": self.symbol,
                "duration": self.duration,
                "duration_unit": self.duration_unit
            }
            if self.take_profit:
                insert_dict(proposal, 'limit_order', {'take_profit': self.take_profit})
            if self.stop_loss:
                insert_dict(proposal, 'limit_order', {'stop_loss': self.stop_loss})
        else:
            return super()._proposal()
        return json.dumps(proposal)

    def check_bid(self, runs=0, show=False, **kwargs) -> None:
        """
        Validates bid price calculations for all proposals over multiple runs.

        The method calculates the correct bid prices based on internal financial 
        models and compares them to the stored or API-provided values.

        Parameters
        ----------
        runs : int, optional
            Number of additional random contracts to validate. Default is 0.
        show : bool, optional
            If True, displays proposals with bid checks that failed. Default is False.
        **kwargs : dict, optional
            Custom parameters to override when generating random contracts.

        Returns
        -------
        None

        Notes
        -----
        - The results are stored in `self.stats['bid_check']` categorized into:
          `total`, `expired`, `sold`, and `bid`.
        """
        for i in range(runs):
            clear_output(wait=True)
            self.buy_contract(random=1, **kwargs)
        self.get_all_poc()
        clear_output(wait=True)
        self.stats['bid_check'] = {
            'total': {'cases': 0, 'pass': 0, 'fail': 0},
            'expired': {'cases': 0, 'pass': 0, 'fail': 0},
            'sold': {'cases': 0, 'pass': 0, 'fail': 0},
            'bid': {'cases': 0, 'pass': 0, 'fail': 0}
        }
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] == 'proposal_open_contract' and 'error' not in item and \
                    item['proposal_open_contract']['contract_type'] in ['CALL', 'CALLE', 'PUT', 'PUTE'] and 'entry_tick' in item['proposal_open_contract']:
                temp_dict = {}
                temp_dict['bid'], temp_dict['bid_check'], temp_dict['close_flag'] = risefall_bid(item)
                insert_dict(item, 'mv', temp_dict)
                self.stats['bid_check'][temp_dict['close_flag']]['cases'] += 1
                self.stats['bid_check']['total']['cases'] += 1
                if item['mv']['bid_check']:
                    self.stats['bid_check'][temp_dict['close_flag']]['pass'] += 1
                    self.stats['bid_check']['total']['pass'] += 1
                else:
                    self.stats['bid_check'][temp_dict['close_flag']]['fail'] += 1
                    self.stats['bid_check']['total']['fail'] += 1
                if not item['mv']['bid_check'] and show:
                    display(item)
        self.stats_bid = dict_to_df(self.stats['bid_check'], 'Bid')
        return self.stats_bid

    def check_payout(self, runs=0, show=False, **kwargs):
        """
        Validates payout accuracy for Rise/Fall contracts.

        This method computes the payout for saved and random contracts and compares 
        it to API-provided values.

        Parameters
        ----------
        runs : int, optional
            Number of additional random contracts to validate. Default is 0.
        show : bool, optional
            If True, displays contracts with failed payout validations. Default is False.
        **kwargs : dict, optional
            Custom parameters to override when generating random contracts.

        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing payout validation statistics.

        Notes
        -----
        Stores the validation results in `self.stats['payout_check']`.
        """
        for i in range(runs):
            clear_output(wait=True)
            self.buy_contract(random=1, **kwargs)
        self.get_all_poc()
        clear_output(wait=True)
        self.stats['payout_check'] = {'cases': 0, 'pass': 0, 'fail': 0}
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] == 'proposal_open_contract' and 'error' not in item and \
                    item['proposal_open_contract']['contract_type'] in ['CALL', 'CALLE', 'PUT', 'PUTE'] and 'entry_tick' in item['proposal_open_contract']:
                temp_dict = {}
                temp_dict['payout'], temp_dict['payout_check'], temp_dict['close_flag'] = risefall_payout(item)
                insert_dict(item, 'mv', temp_dict)
                self.stats['payout_check']['cases'] += 1
                if item['mv']['payout_check']:
                    self.stats['payout_check']['pass'] += 1
                else:
                    self.stats['payout_check']['fail'] += 1
                if not item['mv']['payout_check'] and show:
                    display(item)
        self.stats_payout = dict_to_df(self.stats['payout_check'], 'Payout')
        return self.stats_payout

# def multiplier_barriers(prop) -> float:
#     """
#     Calculates the ask price for a turbo contract.

#     Parameters
#     ----------
#     symbol : str
#         The underlying symbol for contract.
#     spot : float
#         The current spot price of the instrument.
#     barrier : float
#         The barrier level for the contract.
#     duration_unit : str
#         The unit of duration, e.g., 't' for ticks, 'd' for days, etc.
#     phi : int
#         The direction of the contract, 1 for long, -1 for short.

#     Returns
#     -------
#     float
#         The calculated ask price for the turbo contract.
#     """
#     tick_comm = 'tick_comm_' + ('up' if phi == 1 else 'down')
#     tick_comm += '_t' if  duration_unit == 't' else  '_d' if duration_unit == 'd' else '_i'
#     sigma = get_sigma(symbol)
#     is_r = 2 if re.search(re.compile(r'R'), symbol) else 1
#     if phi == 1:
#         commup = BO['turbos'][tick_comm][symbol] * sigma * np.sqrt(is_r/365/86400)
#         return spot * (1 + commup) - barrier
#     else:
#         commdown = BO['turbos'][tick_comm][symbol] * sigma * np.sqrt(is_r/365/86400)
#         return barrier - spot * (1 - commdown) 

def mult_dc_price(prop) -> float:
    """
    Computes the dynamically adjusted cancellation (DC) price for a multiplier contract.

    This function calculates the DC price based on the underlying asset's details, the 
    multiplier, commission, and other financial parameters. It accounts for the contract 
    type (UP/DOWN), barrier adjustments, and relevant financial formulas.

    Parameters
    ----------
    prop : dict
        The proposal data, containing all relevant contract details, such as `multiplier`, 
        `entry_spot`, `current_spot`, `cancellation`, and other properties.

    Returns
    -------
    float
        The dynamically calculated DC price rounded to the appropriate currency precision.

    Notes
    -----
    - The calculation involves barrier adjustments, option pricing models, and spot 
      price movement adjustments.
    - The method also takes into account commission fees and contract multipliers.
    """
    r = 0
    q = 0
    symbol = prop['proposal_open_contract']['underlying']
    sigma = get_sigma(symbol)
    stake = get_stake(prop)
    is_r = 2 if re.search(re.compile(r'R'), symbol) else 1
    dt = is_r/365/86400
    t = (prop['proposal_open_contract']['cancellation']['date_expiry'] - prop['proposal_open_contract']['date_start'])/365/86400    
    try:
        spot = float(prop['proposal_open_contract']['entry_spot'])
    except Exception as e:
        spot = float(prop['proposal_open_contract']['current_spot'])      
    mult = prop['proposal_open_contract']['multiplier']
    comm = BO['multipliers']['commission'][symbol]
    phi = 1 if 'UP' in prop['proposal_open_contract']['contract_type'] else -1
    k = spot * (1 + phi * comm) # barrier adjustment use negative phi in DC as when phi = 1, h<k so phi for barrier adjustment is -1
    h = spot * (1 - phi * (1/mult-comm)) # stop out barrier
    h *= np.exp((1 if h > k else -1)*0.5826*sigma*np.sqrt(dt))
    lambda_ = (r - q + sigma**2/2) / (sigma**2)
    d1 = (np.log(spot/k) + (r - q + sigma**2/2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    x1 = (np.log(spot/h)) / (sigma * np.sqrt(t)) + lambda_ * sigma * np.sqrt(t)
    y1 = (np.log(h/spot)) / (sigma * np.sqrt(t)) + lambda_ * sigma * np.sqrt(t)
    y = (np.log((h**2)/(spot*k))) / (sigma * np.sqrt(t)) + lambda_ * sigma * np.sqrt(t)
    doc, uop = 0, 0
    if phi==1: #down and out call 
        call = spot * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
        dic = spot * np.exp(-q * t) * (h / spot) ** (2 * lambda_) * (norm.cdf(y)) - \
        k * np.exp(-r * t) * (h / spot) ** (2 * lambda_ - 2) * norm.cdf(y - sigma * np.sqrt(t))
        doc = call - dic 
    else:
        put = -spot * np.exp(-q * t) * norm.cdf(-d1) + k * np.exp(-r * t) * norm.cdf(-d2)
        uip = -spot * np.exp(-q * t) * (h / spot) ** (2 * lambda_) * (norm.cdf(-y)) + \
        k * np.exp(-r * t) * (h / spot) ** (2 * lambda_ - 2) * norm.cdf(-y + sigma * np.sqrt(t))
        uop = put - uip
    dc_comm = 1.05
    dc_price = round((doc + uop)*stake*mult/spot * dc_comm, get_currency_precision(prop))
    return dc_price
                                                                                        
def multiplier_payout(prop) -> tuple:
    """
    Calculates the payout and verifies the accuracy of a multiplier contract.

    Based on the contract's details, this function calculates the payout (`bid` price),
    examines whether the actual payout matches the expected payout, and checks for 
    the close flag (e.g., whether the contract has been sold).

    Parameters
    ----------
    prop : dict
        The proposal data as a dictionary.

    Returns
    -------
    tuple
        A tuple containing:
        - bid (float): The computed payout (`bid` price) for the contract.
        - dc_price (float): The dynamically adjusted cancellation price, if applicable.
        - match_flag (bool): A flag indicating whether the calculated payout matches the server's offered payout.
        - close_flag (str): A string indicating whether the contract was 'sold' or is in 'bid' state.

    Notes
    -----
    - The calculation adjusts for commission fees and multiplier factors.
    - If the contract has a `cancellation` property, the dynamically computed DC price
      is validated against the server's provided cancellation price.
    """
    stake = get_stake(prop)
    mult = prop['proposal_open_contract']['multiplier']
    phi = 1 if 'UP' in prop['proposal_open_contract']['contract_type'] else -1
    # display(prop)
    try:
        entry = float(prop['proposal_open_contract']['entry_spot'])
    except Exception as e:
        entry = float(prop['proposal_open_contract']['current_spot'])     
    symbol = prop['proposal_open_contract']['underlying']
    api_bid = prop['proposal_open_contract']['bid_price']
    api_profit =  prop['proposal_open_contract']['profit']
    try:
        spot = float(prop['proposal_open_contract']['exit_tick'])
    except Exception as e:
        spot = float(prop['proposal_open_contract']['current_spot'])
    comm = BO['multipliers']['commission'][symbol]
    bid = (1 + (((phi * (spot - entry)) / entry) - comm) * mult) * stake
    profit = bid - stake
    match_flag = np.round(bid, get_currency_precision(prop)) == api_bid and np.round(bid - stake, get_currency_precision(prop)) == api_profit
    close_flag = 'sold' if 'exit_tick' in prop['proposal_open_contract'] else 'bid'
    dc_price = 0
    if 'cancellation' in prop['proposal_open_contract']:
        dc_price = mult_dc_price(prop)
        match_flag = dc_price == prop['proposal_open_contract']['cancellation']['ask_price'] and match_flag
    return bid, dc_price, match_flag, close_flag

class Multiplier(Product):
    """
    A class to represent a Multiplier product, inheriting from the Product class.

    Attributes
    ----------
    species : str
        The type of the product, always 'Multiplier'.
    contract_type : str
        The type of contract, default is 'MULTUP'.
    growth_rate : float
        The growth rate of the accumulator, default is 0.05.

    Methods
    -------
    randomise_attributes()
        Randomises the attributes of the instance.
    _proposal()
        Generates a JSON-encoded proposal specific to the Multiplier product.
    check_payout(runs=0, show=False, **kwargs)
        Validates payout conditions over a specified number of runs.
    """
    
    species = "Multiplier"
    
    def __init__(self, **kwargs) -> None:
        """
        Initializes a Multiplier instance.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to initialize the instance attributes. Expected keys include:
            - 'growth_rate': (float) The growth rate, defaults to 0.05.
            - 'contract_type': (str) The type of contract, defaults to 'MULTUP'.
            - 'multiplier': (int) The multiplier value, defaults to 10.
            - 'cancellation': (int) Cancellation value, defaults to 0.

        Returns
        -------
        None
        """      
        super().__init__(**kwargs)
        self.contract_type = kwargs.get('contract_type', 'MULTUP')
        self.multiplier = kwargs.get('multiplier', 10)
        self.cancellation = kwargs.get('cancellation', 0)
        self.user_input_params = ['stake', 'symbol', 'multiplier', 'contract_type', 'cancellation']

    def randomise_attributes(self) -> None:
        """
        Randomises the attributes of the Multiplier instance.

        Sets the symbol to a random choice from the available symbols, 
        the stake to a random value within the defined range, and
        the multiplier and cancellation values from predefined lists.

        Returns
        -------
        None
        """
        self.symbol = random.choice(list(symbols.keys()))
        self.stake = round(np.random.uniform(1,BO['accumulators']['max_stake'][self.currency]), BO['currency_precision'][self.currency])
        self.contract_type = random.choice(['MULTUP', 'MULTDOWN'])
        self.multiplier = random.choice(BO['multipliers']['multiplier'][self.symbol])
        self.cancellation = random.choice(BO['multipliers']['cancellation']+[0])

    def _proposal(self) -> str:
        """
        Generates a proposal dictionary specific to the Multiplier product.

        Constructs a proposal dictionary based on the instance's configured attributes. 
        Includes additional parameters like `take_profit` and `stop_loss` if they are set. 
        The proposal is then JSON-encoded.

        Returns
        -------
        str
            A JSON string containing the proposal data.
        """
        proposal = {}
        if self.proposal_type == 'price_proposal':
            proposal = {
                "proposal": 1,
                "basis": 'stake',
                "amount": self.stake,
                "contract_type": self.contract_type,
                "currency": self.currency,
                "symbol": self.symbol,
                "multiplier" : self.multiplier,
            }
            if self.take_profit:
                insert_dict(proposal, 'limit_order', {'take_profit': self.take_profit})
            if self.stop_loss:
                insert_dict(proposal, 'limit_order', {'stop_loss': self.stop_loss})
            if self.cancellation:
                insert_dict(proposal, 'cancellation', self.cancellation)
        else:
            return super()._proposal()
        return json.dumps(proposal)
    
    # def check_barriers(self, runs=0, show=False, **kwargs) -> None:
    #     """
    #     Checks barrier conditions for all saved proposals and the given number of runs.

    #     Randomises attributes and generates price proposals for the specified number of runs.
    #     Analyzes messages to check if barriers are accurate and displays the results.

    #     Parameters
    #     ----------
    #     runs : int
    #         The number of times to run the check.
    #     show : bool
    #         If True, displays messages with failed barrier checks.
    #     **kwargs : dict
    #         Additional keyword arguments for setting attributes.

    #     Returns
    #     -------
    #     None
    #     """
    #     for i in range(runs):
    #         clear_output(wait = True)
    #         self.randomise_attributes()
    #         self._set_attributes(**kwargs)
    #         self.price_proposal()
    #     clear_output(wait = True)
    #     self.accu_barrier_stats = {'cases': 0, 'pass': 0, 'fail': 0}        
    #     for item in self.messages:
    #         if 'msg_type' in item and item['msg_type'] == 'proposal' and 'error' not in item and 'ACCU' in item['echo_req']['contract_type']:
    #             temp_dict = {}
    #             temp_dict['high_barrier'], temp_dict['low_barrier'], temp_dict['barrier_spot_distance'], temp_dict['barrier_check'] = accu_barriers(item)
    #             insert_dict(item, 'mv', temp_dict)
    #             self.accu_barrier_stats['cases'] += 1
    #             if item['mv']['barrier_check']: 
    #                 self.accu_barrier_stats['pass'] += 1
    #             else: 
    #                 self.accu_barrier_stats['fail'] += 1                
    #             if not item['mv']['barrier_check'] and show:
    #                 display(item)

    #     self.stats_barrier = dict_to_df(self.accu_barrier_stats, 'Barriers')   
    #     return self.stats_barrier
        
    def check_payout(self, runs=0, show=False, **kwargs):
        """
        Validates payout conditions for all saved proposals over multiple runs.

        Executes buy contracts with randomized or user-configured attributes for the 
        specified number of runs and analyzes the resulting messages.

        Parameters
        ----------
        runs : int, optional
            The number of iterations to validate payout conditions, defaults to 0.
        show : bool, optional
            If True, displays messages with failed payout checks, defaults to False.
        **kwargs : dict
            Additional keyword arguments to set attributes during the validation process.

        Returns
        -------
        None
        """
        for i in range(runs):
            clear_output(wait = True)
            self.buy_contract(random=1, **kwargs)
        self.get_all_poc() 
        clear_output(wait = True)
        self.stats['payout_check'] = {
            'total': {'cases': 0, 'pass': 0, 'fail': 0},
            'sold': {'cases': 0, 'pass': 0, 'fail': 0},
            'bid': {'cases': 0, 'pass': 0, 'fail': 0}
        }
        for item in self.messages:
            if 'msg_type' in item and item['msg_type'] == 'proposal_open_contract' and 'error' not in item and \
            'MULT' in item['proposal_open_contract']['contract_type']:
                temp_dict = {}
                temp_dict['payout'], temp_dict['cancellation'], temp_dict['payout_check'], temp_dict['close_flag'] = multiplier_payout(item) 
                insert_dict(item, 'mv', temp_dict)
                self.stats['payout_check'][temp_dict['close_flag']]['cases'] += 1
                self.stats['payout_check']['total']['cases'] += 1
                if item['mv']['payout_check']: 
                    self.stats['payout_check'][temp_dict['close_flag']]['pass'] += 1
                    self.stats['payout_check']['total']['pass'] += 1
                else: 
                    self.stats['payout_check'][temp_dict['close_flag']]['fail'] += 1  
                    self.stats['payout_check']['total']['fail'] += 1
                if not item['mv']['payout_check'] and show:
                    display(item) 
        self.stats_payout = dict_to_df(self.stats['payout_check'], 'Payout')
        return self.stats_payout


if __name__ == "__main__":
    run_gui()