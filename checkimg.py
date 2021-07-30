import cv2
root="/home/mathiane/VAE/Normal_tiles.txt"
with open(root, 'r') as f:
    content =  f.readlines()
files_list = []
for x in content:
    x =  x.strip()
    if x.find('reject') == -1:
        files_list.append(x)
with open('ImgSize.txt', 'a') as f:
    for ele in files_list:

        im = cv2.imread(ele)

        if im.shape[0] != 512:
            print(ele, im.shape)
            f.write('{}\t{}\n'.format(ele, str(im.shape)))
