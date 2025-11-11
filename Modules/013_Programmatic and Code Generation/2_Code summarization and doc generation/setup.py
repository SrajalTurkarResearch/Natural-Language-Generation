# setup.py
# Purpose: Installs and checks libraries needed for NLG code summarization and doc generation.
# Why: As a scientist, you need tools ready to experiment, like preparing a lab before tests.
# Run this first to ensure other scripts work.

# Import to check if libraries are installed
import importlib.util
import sys

# List of required libraries
required_libraries = ["transformers", "matplotlib", "pandas"]


def check_and_install(library):
    """Checks if a library is installed; if not, tries to install it."""
    if importlib.util.find_spec(library) is None:
        print(f"{library} not found. Installing...")
        try:
            import pip

            pip.main(["install", library])
            print(f"{library} installed successfully!")
        except Exception as e:
            print(f"Error installing {library}: {e}")
    else:
        print(f"{library} is already installed.")


# Check and install each library
for lib in required_libraries:
    check_and_install(lib)

# Verify installation
print("\nChecking library versions:")
try:
    import transformers

    print(f"Transformers version: {transformers.__version__}")
except:
    print("Transformers not installed properly.")
try:
    import matplotlib

    print(f"Matplotlib version: {matplotlib.__version__}")
except:
    print("Matplotlib not installed properly.")
try:
    import pandas

    print(f"Pandas version: {pandas.__version__}")
except:
    print("Pandas not installed properly.")

print("\nSetup complete! You're ready to run NLG experiments.")
# How this helps: Like setting up a telescope before stargazing, this ensures your tools are ready.
# Next: Run other scripts (e.g., summarization.py) to start learning NLG.
