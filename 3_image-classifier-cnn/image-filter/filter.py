"""
Image Edge Detection Filter

This script applies an edge detection filter to an image using a 3x3 kernel.
The filter enhances edges in the image by detecting rapid changes in pixel intensity.

Summary:
- Applies a Laplacian edge detection kernel to highlight edges
- Uses a 3x3 kernel with values [-1, -1, -1, -1, 8, -1, -1, -1, -1]
- Converts input image to RGB format before processing
- Displays the filtered result

Usage:
    python filter.py <image_filename>
    
    Example:
        python filter.py sample.jpg
        python filter.py /path/to/image.png

Requirements:
    - PIL (Pillow) library
    - Valid image file (jpg, png, etc.)

Output:
    - Displays the edge-detected image in a new window
"""

import math
import sys

from PIL import Image, ImageFilter

# Ensure correct usage
if len(sys.argv) != 2:
    sys.exit("Usage: python filter.py filename")

# Open image
image = Image.open(sys.argv[1]).convert("RGB")

# Filter image according to edge detection kernel
filtered = image.filter(ImageFilter.Kernel(
    size=(3, 3),
    kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
    scale=1
))

# Show resulting image
filtered.show()
