import sys
import logging
import argparse
import importlib.util
import pathlib

# To include this in your config script, run:
# if __name__ == "__main__":
#     from qua_dashboards.core.dashboard_builder import build_dashboard 
#     components_to_run = get_components()
#     app = build_dashboard(components=components_to_run, title="title")
#     app.run_server(debug=True, port=8052, use_reloader=False) 


# Make sure imports within the library work correctly
try:
    from .core.dashboard_builder import build_dashboard
except ImportError:
    # Handle cases where the script might be run directly during development
    from core.dashboard_builder import build_dashboard  # type: ignore

# --- Logging ---
# Configure logging level based on arguments later if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qua_dashboards.cli")


def load_module_from_path(file_path_str: str):
    """Loads a Python module dynamically from a file path."""
    file_path = pathlib.Path(file_path_str).resolve() # Get absolute path
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    module_name = file_path.stem  # Use filename without extension as module name

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
         raise ImportError(f"Could not create module spec for {file_path}")

    config_module = importlib.util.module_from_spec(spec)

    # Add the directory of the config file to sys.path temporarily 
    # to allow imports within the config file (e.g., importing hardware_setup)
    config_dir = str(file_path.parent)
    sys.path.insert(0, config_dir)
    try:
        spec.loader.exec_module(config_module)
    finally:
        # Clean up sys.path
        if sys.path[0] == config_dir:
            sys.path.pop(0)

    return config_module


def run_dashboard_from_config(config_file_path: str, port: int, debug: bool = True, use_reloader: bool = False):
    """
    Loads configuration from a file, builds, and runs the dashboard.

    Args:
        config_file_path: Absolute or relative path to user's configuration python file.
        port: The port number to run the server on.
        debug: Enable Dash debug mode.
        use_reloader: Enable Dash reloader.
    """
    logger.info(f"Attempting to launch dashboard using config: '{config_file_path}'")

    try:
        # Dynamically import the configuration module from the file path
        config_module = load_module_from_path(config_file_path)

        # Check if the required function exists
        if not hasattr(config_module, 'get_components'):
            logger.error(
                f"Configuration file '{config_file_path}' does not define a "
                "'get_components' function."
            )
            sys.exit(1)

        get_components_func = getattr(config_module, 'get_components')

        logger.info("Instantiating components from config...")
        component_instances = get_components_func()

        if not isinstance(component_instances, list) or not component_instances:
            logger.error(
                f"'get_components' function in '{config_file_path}' did not "
                "return a non-empty list."
            )
            sys.exit(1)

        # Determine a title (optional)
        config_name = pathlib.Path(config_file_path).stem.replace('config_', '')
        title = f"Dashboard ({config_name.replace('_', ' ').title()})"

        logger.info("Building dashboard...")
        app = build_dashboard(
            components=component_instances,
            title=title
        )

        logger.info(f"Starting Dash server on port {port}...")
        # Note: use_reloader=False might be important for stability depending on environment
        app.run_server(debug=debug, port=port, use_reloader=use_reloader, host='0.0.0.0') # Host 0.0.0.0 makes it accessible on the network

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file_path}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Failed to import module from '{config_file_path}' or one of its imports: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Failed to configure or launch the dashboard using '{config_file_path}'.")
        sys.exit(1)

def main():
    """Main function executed by the command-line entry point."""
    parser = argparse.ArgumentParser(description="Launch a configured QUA Dashboard.")
    parser.add_argument(
        "config_file", 
        help="Path to the Python configuration file containing the 'get_components' function."
    )
    parser.add_argument(
        "-p", "--port", 
        type=int, 
        default=8050, 
        help="Port number for the Dash server (default: 8050)."
    )
    parser.add_argument(
        "--no-debug", 
        action="store_true", 
        help="Disable Dash debug mode."
    )
    parser.add_argument(
        "--use-reloader", 
        action="store_true", 
        help="Enable Dash auto-reloader (can sometimes cause issues)."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable verbose logging (DEBUG level)."
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")
        # You might want to set levels for other loggers too
        # logging.getLogger('quam_squid_lab').setLevel(logging.DEBUG) 

    run_dashboard_from_config(
        config_file_path=args.config_file, 
        port=args.port, 
        debug=not args.no_debug, 
        use_reloader=args.use_reloader
    )

if __name__ == "__main__":
    # Allows running the cli script directly for development/testing
    # e.g., python -m qua_dashboards.cli ../my_experiment_scripts/config/config_video_mode_random.py
    main()