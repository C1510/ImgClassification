import sys, os, subprocess

# implement pip as a subprocess:
package = 'opencv-python'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package}'])
package = 'pandas'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package}'])

if not os.path.isdir(f'imgs/'):
    os.makedirs(f'imgs/')
if not os.path.isdir(f'imgs_classified/'):
    os.makedirs(f'imgs_classified/')
if not os.path.isdir(f'imgs_classified_png/'):
    os.makedirs(f'imgs_classified_png/')
if not os.path.isdir(f'imgs_rectangled/'):
    os.makedirs(f'imgs_rectangled/')