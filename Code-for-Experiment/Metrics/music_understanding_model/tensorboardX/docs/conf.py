import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.intersphinx',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages']

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'tensorboardX'
copyright = '2017, tensorboardX Contributors'
author = 'tensorboardX Contributors'
version = ''
release = ''
language = None

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False

html_theme = 'sphinx_rtd_theme'
htmlhelp_basename = 'tensorboardXdoc'

latex_elements = {}
latex_documents = [
    (master_doc, 'tensorboardX.tex', 'tensorboardX Documentation',
     'tensorboardX Contributors', 'manual'),
]

man_pages = [
    (master_doc, 'tensorboardX', 'tensorboardX Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'tensorboardX', 'tensorboardX Documentation',
     author, 'tensorboardX', 'One line description of project.',
     'Miscellaneous'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'torch': ('http://pytorch.org/docs/master', None),
    'matplotlib': ('http://matplotlib.sourceforge.net/', None),
}