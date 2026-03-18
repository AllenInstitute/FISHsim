""" Load defaults and environment variables for the FISHsim package. """

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os

# Load environment variables
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
    print(f"Using env file {dotenv_path}")
except OSError:
    dotenv_path = (
        Path.home().joinpath("fishsim.env")
    )
    print(f"Using env file {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

# Define configuration variables
RESOURCES_DIR = os.getenv("RESOURCES_DIR", default=str(Path(__file__).parent.parent.parent.joinpath("resources")))
CODEBOOK_DIR = os.getenv("CODEBOOK_DIR", default=str(Path(RESOURCES_DIR).joinpath("codebooks")))
RESULTS_DIR = os.getenv("RESULTS_DIR", default=str(Path.home()))