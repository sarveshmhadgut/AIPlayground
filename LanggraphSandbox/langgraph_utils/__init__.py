from rich.console import Console
from rich.theme import Theme


def get_console() -> Console:
    """Create and return a Rich Console configured with the project's standard theme."""
    theme = Theme(
        {
            "json.key": "#c1a2ff",
            "json.string": "#FCCEA1",
            "json.number": "#A1C4FD",
            "json.boolean": "#A6F5D8",
            "json.null": "#ffb3ba",
        }
    )
    return Console(
        theme=theme,
        force_terminal=True,
        force_jupyter=False,
        color_system="truecolor",
        width=120,
    )


console = get_console()
