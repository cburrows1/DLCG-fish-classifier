import os
from natsort import natsorted 

path = 'dataset'

image_paths = []
for subdir, dirs, files in os.walk(path):
    for filename in files:
        filepath = subdir + os.sep + filename
        filepath_lower = filepath.lower()
        if filepath_lower.endswith(".jpg") or filepath_lower.endswith(".png"):
            image_paths.append(filepath)
        elif os.path.isfile(os.path.abspath(filepath)):
            os.remove(filepath)
            print("removed: " + filepath)


def get_file_parts(path):
    fname = os.path.splitext(path)[0]
    file_parts = os.path.splitext( fname )
    number = file_parts[1].replace('.','')
    name = file_parts[0]
    return name, int(number)

image_paths = natsorted(image_paths)
last_name = ""
for img in image_paths:
    curr_name, curr_num = get_file_parts(img)
    if last_name != curr_name:
        last_name = curr_name
        last_num = -1
    if last_num + 1 != curr_num:
        os.rename(img,img.replace(str(curr_num),str(last_num + 1)))
        print("found: " + img)
    last_num += 1
    os.rename(img, img.lower())