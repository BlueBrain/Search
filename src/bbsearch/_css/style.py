"""CSS styling utilities."""

import pathlib


def get_css_style():
    """Get content of CSS style sheet."""
    css_file = pathlib.Path(__file__).parents[0] / 'stylesheet.css'
    with open(css_file, 'r') as f:
        css_style = f.read()
    return css_style
