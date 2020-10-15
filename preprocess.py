import os
from shutil import copyfile

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径





basepath = "D:\\WorkSpace\\FaceRecognition\\CelebA\\training_data"
imgbase = "D:\\WorkSpace\\FaceRecognition\\CelebA\\img_align_celeba"
f= open("D:\\WorkSpace\\FaceRecognition\\CelebA\\identity_CelebA.txt")
for line in f:
    content = line.split()
    dispath = os.path.join(basepath,content[1],content[0])
    imgpath = os.path.join(imgbase,content[0])
    copyfile(imgpath,dispath)
print("done!")
