import os
from natsort import natsorted 

# set paths of original dataset and path to where testset should go (note: all required directories must exist in testset already as they do in dataset)
path = 'dataset'
testset = 'testset'

# boolean to enable moving of file (set to false to see debug results without modifying files)
move = True

#split fish file path into parts
def get_file_parts(path):
    fname = os.path.splitext(path)[0]
    file_parts = os.path.splitext( fname )
    number = file_parts[1].replace('.','')
    name = file_parts[0]
    try:
        number_int = int(number)
    except ValueError:
        print("Invalid int at: " + path)
    return name, int(number)

image_paths = []
for subdir, dirs, files in os.walk(path):
    for filename in files:
        filepath = subdir + os.sep + filename
        image_paths.append(filepath)


# sort all image file paths
image_paths = natsorted(image_paths)
fish_group_paths = {}
for img in image_paths:
    name, num = get_file_parts(img)
    if name in fish_group_paths:
        fish_group_paths[name].append(img)
    else:
        fish_group_paths[name] = [img]

# take the last two images of each fish and move to test set
for group in fish_group_paths:
    print(group + ": ")
    train = fish_group_paths[group][:-2]
    test = fish_group_paths[group][-2:]
    print("\tTrain:")
    for fish in train:
        print("\t\t" + fish)
    print("\tTest:")
    for fish in test:
        new_name = os.path.join(testset,'/'.join(fish.split("/")[1:]))
        print("\t\t{} => {}".format(fish,new_name))
        if move:
            os.rename(fish, new_name)