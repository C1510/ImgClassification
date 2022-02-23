import sys, os, subprocess

# implement pip as a subprocess:
def get_package(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package}'])

get_package('opencv-python')
get_package('pandas')
get_package('torch')
get_package('torchvision')
get_package('torchaudio')


if not os.path.isdir(f'imgs/'):
    os.makedirs(f'imgs/')
if not os.path.isdir(f'imgs_classified/'):
    os.makedirs(f'imgs_classified/')
if not os.path.isdir(f'imgs_classified_png/'):
    os.makedirs(f'imgs_classified_png/')
if not os.path.isdir(f'imgs_rectangled/'):
    os.makedirs(f'imgs_rectangled/')
if not os.path.isdir(f'neural_network/models/'):
    os.makedirs(f'neural_network/models/')
if not os.path.isdir(f'neural_network/classified_png/'):
    os.makedirs(f'neural_network/classified_png/')