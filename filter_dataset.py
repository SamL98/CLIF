from os import listdir, remove
from os.path import join

img_dir = 'E:/lerner/clif/img'
gt_dir = 'E:/lerner/clif/gt'

imgs = listdir(img_dir)
gts = listdir(gt_dir)
crop_names = [gt[:gt.index('.')] for gt in gts]

for img_name in imgs:
	if not img_name[:img_name.index('.')] in crop_names:
		remove(join(img_dir, img_name))