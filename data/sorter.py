import mat4py
import os
import shutil

data = mat4py.loadmat('data/cars196/car_devkit/devkit/cars_train_annos.mat')

classes = data['annotations'].get('class')

images = data['annotations'].get('fname')

root_name = "data/images"

images_path = "data/cars196/cars_train/cars_train"

if not os.path.exists(root_name):
    os.makedirs(root_name)

for i in range(1,197):
    path = os.path.join(root_name, str(i))
    
    if not os.path.exists(path):
        os.mkdir(path)

for i in range(len(images)):
    old_path = os.path.join(images_path, images[i])
    new_path = os.path.join(root_name, str(classes[i]), images[i])

    shutil.move(old_path, new_path)

    if i % 500 == 0:
        print(f"Moved {i} images")

print(f"Moved {len(images)} images")