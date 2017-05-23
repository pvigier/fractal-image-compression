import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import math

# Manipulate channels

def get_greyscale_image(img):
	return np.mean(img[:,:,:2], 2)

def extract_rgb(img):
	return img[:,:,0], img[:,:,1], img[:,:,2]

def assemble_rbg(img_r, img_g, img_b):
	shape = (img_r.shape[0], img_r.shape[1], 1)
	return np.concatenate((np.reshape(img_r, shape), np.reshape(img_g, shape), 
		np.reshape(img_b, shape)), axis=2)

# Transforms

def reduce(img, factor):
	result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			result[i,j] += np.mean(img[i*factor:(i+1)*factor,j*factor:(j+1)*factor])
	return result

def rotate(img, angle):
	return ndimage.rotate(img, angle, reshape=False)

def flip(img, direction):
	return img[::direction,:]

def apply_transform(img, direction, angle, contrast=1.0, brightness=0.0):
	return contrast*rotate(flip(img, direction), angle) + brightness

# Contrast and brightness

def find_contrast_and_brightness1(D, S):
	# Fix the contrast and only fit the brightness
	contrast = 0.75
	brightness = (np.sum(D - contrast*S)) / D.size
	return contrast, brightness 

def find_contrast_and_brightness2(D, S):
	# Fit the contrast and the brightness
	A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
	b = np.reshape(D, (D.size,))
	x, _, _, _ = np.linalg.lstsq(A, b)
	return x[1], x[0]

# Compression for greyscale images

def compress(img):
	transforms = []
	for i in range(img.shape[0] // destination_size):
		transforms.append([])
		for j in range(img.shape[1] // destination_size):
			print(i, j)
			transforms[i].append(None)
			min_d = float('inf')
			# Extract the destination block
			D = img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size]
			for k in range(img.shape[0] // step):
				for l in range(img.shape[1] // step):
					# Extract the source block and reduce it to the shape of D
					S = reduce(img[k*step:k*step+source_size,l*step:l*step+source_size], factor)
					# Test all possible transforms and take the best one
					for direction, angle in candidates:
						T = apply_transform(S, direction, angle)
						contrast, brightness = find_contrast_and_brightness2(D, T)
						T = contrast*T + brightness
						d = np.sum(np.square(D - T))
						if d < min_d:
							min_d = d
							transforms[i][j] = (k, l, direction, angle, contrast, brightness)
	return transforms

def uncompress(transforms, nb_iter=8):
	height = len(transforms) * destination_size
	width = len(transforms[0]) * destination_size
	iterations = [np.random.randint(0, 256, (height, width))]
	cur_img = np.zeros((height, width))
	for i_iter in range(nb_iter):
		print(i_iter)
		for i in range(len(transforms)):
			for j in range(len(transforms[i])):
				# Apply transform
				k, l, flip, angle, contrast, brightness = transforms[i][j]
				S = reduce(iterations[-1][k*step:k*step+source_size,l*step:l*step+source_size], factor)
				D = apply_transform(S, flip, angle, contrast, brightness)
				cur_img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size] = D
		iterations.append(cur_img)
		cur_img = np.zeros((height, width))
	return iterations

# Compression for color images

def reduce_rgb(img, factor):
	img_r, img_g, img_b = extract_rgb(img)
	img_r = reduce(img_r, factor)
	img_g = reduce(img_g, factor)
	img_b = reduce(img_b, factor)
	return assemble_rbg(img_r, img_g, img_b)

def compress_rgb(img):
	img_r, img_g, img_b = extract_rgb(img)
	return [compress(img_r), compress(img_g), compress(img_b)]

def uncompress_rgb(transforms, nb_iter=8):
	img_r = uncompress(transforms[0], nb_iter)[-1]
	img_g = uncompress(transforms[1], nb_iter)[-1]
	img_b = uncompress(transforms[2], nb_iter)[-1]
	return assemble_rbg(img_r, img_g, img_b)

# Plot

def plot_iterations(iterations):
	# Configure plot
	plt.figure()
	nb_row = math.ceil(np.sqrt(len(iterations)))
	nb_cols = nb_row
	# Plot
	for i, img in enumerate(iterations):
		print(img.shape)
		plt.subplot(nb_row, nb_cols, i+1)
		plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
		plt.title(str(i))# + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(img - cur_img)))) + ')')
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
	plt.tight_layout()

# Parameters

directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = list(zip(directions, angles))
source_size = 8
destination_size = 4
factor = source_size / destination_size
step = source_size

# Tests

def test_greyscale():
	img = mpimg.imread('monkey.gif')
	img = get_greyscale_image(img)
	img = reduce(img, 4)
	plt.figure()
	plt.imshow(img, cmap='gray', interpolation='none')
	transforms = compress(img)
	iterations = uncompress(transforms)
	plot_iterations(iterations)
	plt.show()

def test_rgb():
	img = mpimg.imread('lena.gif')
	img = reduce_rgb(img, 8)
	plt.imshow(np.array(img).astype(np.uint8), interpolation='none')
	transforms = compress_rgb(img)
	retrieved_img = uncompress_rgb(transforms)
	plt.figure()
	plt.imshow(retrieved_img.astype(np.uint8), interpolation='none')
	plt.show()
					
if __name__ == '__main__':
	test_greyscale()
	#test_rgb()