import os
import re

# Set your main module folder containing subfolders
module_folder = "codes"

# Traverse all subfolders in the module folder in sorted order
for subfolder in sorted(os.listdir(module_folder)):
    subfolder_path = os.path.join(module_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # List all .ipynb files that start with numbers in the subfolder
        files = [
            f
            for f in os.listdir(subfolder_path)
            if f.endswith(".ipynb") and re.match(r"\d+_.*\.ipynb$", f)
        ]

        for old_name in files:
            match = re.match(r"(\d+)(_.*)", old_name)
            if match:
                number = int(match.group(1))
                rest = match.group(2)
                new_name = f"{number:03d}{rest}"

                old_path = os.path.join(subfolder_path, old_name)
                new_path = os.path.join(subfolder_path, new_name)

                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed in {subfolder}: {old_name} → {new_name}")
