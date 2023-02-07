#
# galpy documentation build configuration file, created by
# sphinx-quickstart on Sun Jul 11 15:58:27 2010.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import datetime
import importlib
import inspect
import os
import re
import subprocess
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.append(os.path.abspath('.'))

# -- General configuration -----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax','sphinx.ext.ifconfig',
              'sphinx.ext.linkcode',"sphinxext.opengraph"]

# from disnake via:
# https://twitter.com/readthedocs/status/1541830907082022913?s=20&t=eJ293FfjILT7sIxEyz834w
_galpy_module_path = os.path.dirname(importlib.util.find_spec("galpy").origin)
github_repo = "https://github.com/jobovy/galpy"
# Current git reference. Uses branch/tag name if found, otherwise uses commit hash
def git(*args):
    return subprocess.check_output(["git", *args]).strip().decode()
git_ref= None
try:
    git_ref= git("name-rev", "--name-only", "--no-undefined", "HEAD")
    git_ref= re.sub(r"^(remotes/[^/]+|tags)/", "", git_ref)
except Exception:
    pass
# (if no name found or relative ref, use commit hash instead)
if not git_ref or re.search(r"[\^~]", git_ref):
    try:
        git_ref = git("rev-parse", "HEAD")
    except Exception:
        git_ref = "main"
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    try:
        obj= sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj= getattr(obj, part)
        obj= inspect.unwrap(obj)

        if isinstance(obj, property):
            obj= inspect.unwrap(obj.fget)

        path= os.path.relpath(inspect.getsourcefile(obj),start=_galpy_module_path)
        src, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    path = f"{path}#L{lineno}-L{lineno + len(src) - 1}"
    return f"{github_repo}/blob/{git_ref}/galpy/{path}"

ogp_site_name= "galpy documentation"
ogp_site_url= (f"https://docs.galpy.org/en/{git_ref}/"
               if not git_ref == 'main'
               else "https://docs.galpy.org/en/latest/")

ogp_social_cards = {
    "enable": True,
    "image": "images/galpy-logo-small.png",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'galpy'
copyright = f'2010 - {datetime.datetime.now().year}, Jo Bovy'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '1.8.2.dev0'
# The full version, including alpha/beta/rc tags.
release = version
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    version= 'v'+version
    release= 'v'+release
def setup(app):
    app.add_config_value('not_on_rtd',True,True)
    app.add_css_file("custom.css")
    app.add_css_file("gallery.css")
    app.add_js_file("clipboard.min.js")
    app.add_js_file("copycode.js")
    app.add_js_file("papers-gallery.js")
    app.add_js_file("try-galpy.js")
not_on_rtd= not on_rtd

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = []

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = 'sphinxdoc'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = 'images/galpy-logo-small.gif'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'images/logo-small.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
html_use_modindex = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'galpydoc'


# -- Options for LaTeX output --------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'galpy.tex', 'galpy Documentation',
   'Jo Bovy', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
#latex_preamble = ''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True
