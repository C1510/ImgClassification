from PIL import Image
import os, sys
import numpy as np

def tif_to_imgs(tif_folder, data_folder_out):

    tif_files = [os.path.join(tif_folder, o) for o in os.listdir(tif_folder) if
                 os.path.isfile(os.path.join(tif_folder, o))]
    if not os.path.exists(data_folder_out):
        os.mkdir(data_folder_out)
    for count, file_path in enumerate(tif_files):
        if not os.path.exists(f'{data_folder_out}/class_{count}'):
            os.mkdir(f'{data_folder_out}/class_{count}')
        print("The selected stack is a .tif")
        dataset = Image.open(file_path)
        for i in range(dataset.n_frames):
           dataset.seek(i)
           im = Image.fromarray(np.array(dataset))
           im.save(f'{data_folder_out}/class_{count}/img_{i}.png')

tif_to_imgs('data/tiffs','data_2')