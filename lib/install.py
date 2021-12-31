import sys
import subprocess

# implement pip as a subprocess:
package = 'opencv-python'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package}'])