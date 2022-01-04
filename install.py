import sys, os, subprocess

# implement pip as a subprocess:
def get_package(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package}'])

get_package('opencv-python')
get_package('json')
get_package('pandas')

if not os.path.isdir(f'imgs/'):
    os.makedirs(f'imgs/')
if not os.path.isdir(f'imgs_classified/'):
    os.makedirs(f'imgs_classified/')
if not os.path.isdir(f'imgs_classified_png/'):
    os.makedirs(f'imgs_classified_png/')
if not os.path.isdir(f'imgs_rectangled/'):
    os.makedirs(f'imgs_rectangled/')