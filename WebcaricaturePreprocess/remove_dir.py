import os
import shutil


def remove_dir(filelist):
    eyes_dir = "D:\WorkSpace\WebCaricature\eyes_6"
    mouth_dir = "D:\WorkSpace\WebCaricature\mouths_6"
    face_dir ="D:\WorkSpace\WebCaricature\\alignedImages_6"
    with open(filelist) as fp:
        for line in fp:
            dir_eyes = os.path.join(eyes_dir,line.rstrip('\n').rstrip('\t'))
            dir_mouths = os.path.join(mouth_dir, line.rstrip('\n').rstrip('\t'))
            dir_faces = os.path.join(face_dir, line.rstrip('\n').rstrip('\t'))
            try:
                shutil.rmtree(dir_eyes)    #递归删除文件夹
                shutil.rmtree(dir_mouths)
                shutil.rmtree(dir_faces)
            except:
                pass


remove_dir("D:\WorkSpace\WebCaricature\\toremove.txt")