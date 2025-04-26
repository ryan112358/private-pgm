# sphinx-apidoc -o . ../src/mbi
# sphinx-build -b html . _build/html

import sys
import os

sys.path.insert(0, os.path.abspath('../src'))

project = 'private-pgm'
copyright = '2025, Ryan McKenna'
author = 'Ryan McKenna'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon', 
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'myst_parser',
    'sphinx.ext.doctest',
]

# Add this line if it doesn't exist, or ensure it's True if it does
autosummary_generate = True

# Configure autodoc settings
autodoc_typehints = "signature"
autoclass_content = "both"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
