# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'API Testing Automation Project'
copyright = '2024, Matthew-Deriv, VishalMenon-Deriv'
author = 'Matthew-Deriv, VishalMenon-Deriv'
release = '0.0.1'

# file path 
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# order documentation by source code
autodoc_member_order = 'bysource'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__, __setattr__, _set_attributes, _set_app_details, _get_app_markup, _get_app_id, _proposal, _send, _check_connection, _check_closed, _recv_msg_handler',
    'private-members': True,
    'inherited-members': True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]