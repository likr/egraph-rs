# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'egraph'
copyright = f'{datetime.now().year}, Yosuke Onoue'
author = 'Yosuke Onoue'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autoclass_content = 'both'

# -- Options for doctest -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html#configuration

doctest_global_setup = '''
import egraph as eg
import networkx as nx
'''

doctest_test_doctest_blocks = 'default'
doctest_default_flags = 0

# -- Options for intersphinx -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = f'{project} documentation'
html_logo = None
html_favicon = None

# -- Options for Napoleon ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None
