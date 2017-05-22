"""
__init__.py - provides access to the YANN module without requiring installation
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
