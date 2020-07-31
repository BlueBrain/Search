"""CSS styling utilities."""

import pathlib


def get_css_style():
    """Get content of CSS style sheet."""
    css_file = pathlib.Path(__file__).parents[0] / 'stylesheet.css'
    return css_file
