"""CSS styling utilities."""
import pkg_resources


def get_css_style():
    """Get content of CSS style sheet."""
    css_file = pkg_resources.resource_filename(__name__, "stylesheet.css")
    with open(css_file, "r") as f:
        css_style = f.read()
    return css_style
