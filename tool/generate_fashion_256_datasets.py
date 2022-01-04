import os
import shutil
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    new_root = '../fashion_data_256'
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    train_root = '../fashion_data_256/train'
    train_iuv_root = '../fashion_data_256/train_iuv'
    os.makedirs(train_iuv_root, exist_ok=True)
    if not os.path.exists(train_root):
        os.mkdir(train_root)

    test_root = '../fashion_data_256/test'
    test_iuv_root = '../fashion_data_256/test_iuv'
    os.makedirs(test_iuv_root, exist_ok=True)
    if not os.path.exists(test_root):
        os.mkdir(test_root)

    train_images = []
    train_f = open('../fashion_data_256/train.lst', 'r')
    for lines in train_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            train_images.append(lines)

    test_images = []
    test_f = open('../fashion_data_256/test.lst', 'r')
    for lines in test_f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            test_images.append(lines)

    # print(train_images, test_images)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                path_names = path.split('/')[1:]
                # path_names[2] = path_names[2].replace('_', '')
                path_names[3] = path_names[3].replace('_', '')
                path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
                path_names = "".join(path_names)
                # new_path = os.path.join(root, path_names)
                img = Image.open(path)

                # imgcrop = img.crop((40, 0, 216, 256))
                if path_names in train_images and os.path.exists(path.replace('.jpg', '_IUV.png')):
                    img_iuv = Image.open(path.replace('.jpg', '_IUV.png'))
                    img.save(os.path.join(train_root, path_names))
                    img_iuv.save(os.path.join(train_iuv_root, path_names.replace('.jpg', '.png')))
                elif path_names in test_images and os.path.exists(path.replace('.jpg', '_IUV.png')):
                    img_iuv = Image.open(path.replace('.jpg', '_IUV.png'))
                    img.save(os.path.join(test_root, path_names))
                    img_iuv.save(os.path.join(test_iuv_root, path_names.replace('.jpg', '.png')))


make_dataset('../fashion')
# make_dataset('/home/nas1_temp/dataset/deepfashion/Img/img_highres')