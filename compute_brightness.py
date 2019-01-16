from load_data import _DS_INFO, load_image

means = []

n = 1
mean = 0

for i in range(1, _DS_INFO['num_train']+1):
	mean += (load_image('train', i).mean() - mean)/n
	n += 1
	
for i in range(1, _DS_INFO['num_val']+1):
	mean += (load_image('val', i).mean() - mean)/n
	n += 1
	
print(mean)