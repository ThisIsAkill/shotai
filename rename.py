import os
import re

# Directory where the images are located
image_dir = '/home/spike/Documents/GitHub/shotsai/datasets/closeup'

# Regular expression to match image files and capture their extensions
image_regex = re.compile(r'^(.*?)(\.\w+)$')

# List all files in the directory
files = os.listdir(image_dir)

# Counter for naming images
counter = 1

# Iterate through files and rename them
for filename in files:
    match = image_regex.match(filename)
    if match:
        # Get the file extension
        extension = match.group(2)
        # New filename
        new_filename = f'image{counter}{extension}'
        # Paths to old and new files
        old_file = os.path.join(image_dir, filename)
        new_file = os.path.join(image_dir, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        # Increment the counter
        counter += 1

print("Images renamed successfully.")
