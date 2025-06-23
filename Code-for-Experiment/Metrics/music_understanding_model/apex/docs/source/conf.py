import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import apex
import sphinx_rtd_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
]

napoleon_use_ivar = True

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'Apex'
copyright = '2018'
author = 'Christian Sarofeen, Natalia Gimelshein, Michael Carilli, Raul Puri'
version = '0.1'
release = '0.1.0'
language = None
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = True

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

html_static_path = ['_static']

html_context = {
    'css_files': [
        'https://fonts.googleapis.com/css?family=Lato',
        '_static/css/pytorch_theme.css'
    ],
}

htmlhelp_basename = 'PyTorchdoc'

latex_elements = {}

latex_documents = [
    (master_doc, 'apex.tex', 'Apex Documentation',
     'Torch Contributors', 'manual'),
]

man_pages = [
    (master_doc, 'Apex', 'Apex Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'Apex', 'Apex Documentation',
     author, 'Apex', 'One line description of project.',
     'Miscellaneous'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
}

from docutils import nodes
from sphinx.util.docfields import TypedField
from sphinx import addnodes

def patched_make_field(self, types, domain, items, **kw):
    def handle_item(fieldarg, content):
        par = nodes.paragraph()
        par += addnodes.literal_strong('', fieldarg)
        if fieldarg in types:
            par += nodes.Text(' (')
            fieldtype = types.pop(fieldarg)
            if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                typename = u''.join(n.astext() for n in fieldtype)
                typename = typename.replace('int', 'python:int')
                typename = typename.replace('long', 'python:long')
                typename = typename.replace('float', 'python:float')
                typename = typename.replace('type', 'python:type')
                par.extend(self.make_xrefs(self.typerolename, domain, typename,
                                           addnodes.literal_emphasis, **kw))
            else:
                par += fieldtype
            par += nodes.Text(')')
        par += nodes.Text(' -- ')
        par += content
        return par

    fieldname = nodes.field_name('', self.label)
    if len(items) == 1 and self.can_collapse:
        fieldarg, content = items[0]
        bodynode = handle_item(fieldarg, content)
    else:
        bodynode = self.list_type()
        for fieldarg, content in items:
            bodynode += nodes.list_item('', handle_item(fieldarg, content))
    fieldbody = nodes.field_body('', bodynode)
    return nodes.field('', fieldname, fieldbody)

TypedField.make_field = patched_make_field