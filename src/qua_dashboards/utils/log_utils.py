import sys
import logging

__all__ = ["setup_logging"]


def setup_logging(
    name: str | None = None, level=logging.DEBUG, filter_common_libs=True
) -> logging.Logger:
    """Configures logging for the application.

    Args:
        name: The name of the logger to use.
            If None, the root logger will be used.
        level: The logging level to use.
        filter_common_libs: Whether to filter common libraries from logging.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    if filter_common_libs:
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("hpack").setLevel(logging.WARNING)

    if name is not None:
        return logging.getLogger(name)
    else:
        return logging.getLogger("qua_dashboards")
