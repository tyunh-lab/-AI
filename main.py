#cnn を使うには、CUDA 9/cudnn7 環境で実行してください
import face_recognition
import shutil
import os
import random
from matplotlib import pyplot as plt

if False == os.path.exists("manypeople/"):
  os.mkdir("manypeople/")

if False == os.path.exists("noface/"):
  os.mkdir("noface")

# 保存されている人物の顔の画像を読み込む。
known_face_imgs_from = []
curdir = os.listdir(".")#ファイル指定
for name in curdir:
  root, ext = os.path.splitext(name)
  if ext == ".jpg":
    known_face_imgs_from.append(name)
known_face_imgs = []
for path in known_face_imgs_from:
  img = face_recognition.load_image_file(path)
  known_face_imgs.append(img)
    
while len(known_face_imgs_from) != 0:
  # 認証する人物の顔の画像を読み込む。
  people_num = random.randint(0,len(known_face_imgs_from)-1)
  people = known_face_imgs_from[people_num]
  face_img_to_check = face_recognition.load_image_file(people)

  # 顔の画像から顔の領域を検出する。
  known_face_locs = []
  i = 0
  for img in known_face_imgs:
    loc = face_recognition.face_locations(img, model="hog")
    known_face_locs.append(loc)
    i = i + 1
    print("\r"+str(i)+"/"+str(len(known_face_imgs_from)),end="")
  
  face_loc_to_check = face_recognition.face_locations(face_img_to_check, model="hog")#cnn,hog

  #顔が複数あった場合のエラー回避
  known_face_locs1 = []
  known_face_imgs1 = []
  known_face_imgs_from1 = []
  i = 0
  while i != len(known_face_imgs):
    if len(known_face_locs[i]) == 1:#顔が複数検出されていない
      known_face_locs1.append(known_face_locs[i])
      known_face_imgs1.append(known_face_imgs[i])
      known_face_imgs_from1.append(known_face_imgs_from[i])
    elif len(known_face_locs[i]) == 0:
      shutil.move(known_face_imgs_from[i], "noface/")
    i = i + 1
  
  i = 0

  # 顔の領域から特徴量を抽出する。
  known_face_encodings = []
  for img, loc in zip(known_face_imgs1, known_face_locs1):
    (encoding,) = face_recognition.face_encodings(img, loc)
    known_face_encodings.append(encoding)
    if len(face_loc_to_check) == 1:
      (face_encoding_to_check,) = face_recognition.face_encodings(face_img_to_check, face_loc_to_check)

  if len(face_loc_to_check) == 1 and  ("face_encoding_to_check" in locals() or "face_encoding_to_check" in globals()):
    # 抽出した特徴量を元にマッチングを行う。
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check,0.5)#3番目は閾値
    # 各画像との近似度を表示する。
    dists = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    #親の画像を表示
    img = plt.imread(people)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    #ファイル移動
    dir = input("この人の名前は？:")+"/"
    os.mkdir(dir)
    while(i != len(matches)):
      if matches[i] == True:
        shutil.move(known_face_imgs_from1[i], dir)
      i = i + 1
  else:
    shutil.move(known_face_imgs_from[people_num],"manypeople/")

  # 保存されている人物の顔の画像を読み込む。
  known_face_imgs_from = []
  curdir = os.listdir(".")#ファイル指定 
  for name in curdir:
    root, ext = os.path.splitext(name)
    if ext == ".jpg":
      known_face_imgs_from.append(name)

  known_face_imgs = []
  for path in known_face_imgs_from:
    img = face_recognition.load_image_file(path)
    known_face_imgs.append(img)

print("終了しました。")
