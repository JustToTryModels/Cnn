import streamlit as st
import importlib
import pkg_resources

# List of packages you want to check
packages = [
    "streamlit",
    "tensorflow",
    "numpy",
    "Pillow",
    "requests",
    "matplotlib"
]

st.title("📦 Package Version Checker")

st.write("This app displays the installed versions of the required Python packages.")

for pkg in packages:
    try:
        # Import the module
        module = importlib.import_module(pkg if pkg != "Pillow" else "PIL")
        # Get version
        version = pkg_resources.get_distribution(pkg).version
        st.success(f"{pkg} version: {version}")
    except ImportError:
        st.error(f"{pkg} is not installed.")
    except Exception as e:
        st.warning(f"Could not check {pkg}: {e}")
