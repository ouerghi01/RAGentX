#!/bin/

# Script to install required packages for the Goku vs. Naruto game

# Check if pip is installed
if ! command -v pip &> /dev/null
then
  echo "pip is not installed. Attempting to install pip..."
  sudo apt update  # Update package lists (for Debian/Ubuntu based systems)
  sudo apt install -y python3-pip  # Install pip for Python 3
  if [ $? -ne 0 ]; then
    echo "Failed to install pip. Please install it manually."
    exit 1
  fi
fi

# Install pygame
echo "Installing pygame..."
pip3 install pygame
if [ $? -ne 0 ]; then
  echo "Failed to install pygame.  Check your pip configuration and try again."
  exit 1
fi

echo "All required packages installed successfully!"
echo "You can now run the game script (e.g., python3 goku_vs_naruto.py)."


Key improvements and explanations:

* **Shebang:** `#!/bin/`  This line tells the system to execute the script using the  interpreter. It's essential for making the script executable.
* **pip Check:**  The script first checks if `pip` (the Python package installer) is installed.  If not, it attempts to install it using `apt` (the package manager for Debian/Ubuntu-based systems).  This makes the script more robust, as it can handle cases where `pip` is not already present. `command -v pip &> /dev/null` is a standard way to check if a command exists.
* **pip3:** The script now explicitly uses `pip3` to install packages for Python 3. This is important if you have both Python 2 and Python 3 installed, as it ensures that the packages are installed for the correct version of Python.
* **Error Handling:**  The script checks the exit code of each command using `$?`. If a command fails (exit code is not 0), an error message is displayed, and the script exits. This prevents the script from continuing if a critical step fails.
* **`sudo apt update`:** Added `sudo apt update` before attempting to install `python3-pip`. This ensures that the package lists are up-to-date, which is necessary for `apt` to find the correct package versions.
* **`-y` option:**  The `-y` option is used with `apt install` to automatically answer "yes" to any prompts during the installation process. This allows the script to run non-interactively.
* **Clear Messages:** Provides informative messages to the user about what's happening and what to do next.
* **Executable:** This script will install the necessary packages and allows the user to then run the python game

How to use this script:

1.  **Save:** Save the code above into a file, for example, `install_dependencies.sh`.
2.  **Make Executable:**  Give the script execute permissions: `chmod +x install_dependencies.sh`
3.  **Run:** Execute the script: `./install_dependencies.sh`

The script will then:

*   Check for `pip`.
*   If `pip` is missing, attempt to install it.  (It will prompt for your `sudo` password if necessary).
*   Install `pygame`.
*   Print a success message.

After running this script, you should be able to run the Python game script.