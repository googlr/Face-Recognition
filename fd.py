# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
from matplotlib import pylab as plt
import math
# import cv2
import imageio
from numpy import linalg as LA

# Why here dimension is 195*231 while array is 231*195
row, col = 231,195 # Size of the image
file_path = "./FaceDataset/subject"
file_train = [ "01", "02", "03", "07", "10", "11", "14", "15"]
train_size = len(file_train)
# Read file from raw
# face_from_raw = np.fromfile('face.raw', dtype=np.uint8)
# Read file from jpg
# misc.imread()
def plt_face(x):
	plt.imshow(x.reshape((row, col)), cmap=plt.cm.gray)
	plt.xticks([])
	plt.show()

##############################################################################
# Eigenfaces: Training
train_data = np.zeros((row*col, train_size), dtype=np.uint8)
for i in range(0, train_size):
	filename = file_path + file_train[i] + ".normal.jpg"
	train = imageio.imread(filename)
	# test_image = np.fromfile(filename,dtype='uint8',sep="")
	# print(train.shape)
	# print(train.dtype)
	# plt_face(train)
	train_data[:,i] = train.reshape((row*col))


# print(train_data.shape)
# print(train_data.dtype)
mean_face = np.mean(train_data, axis=1).reshape(row*col,1)
# print(mean_face.shape)
# plt_face(mean_face)
M = np.tile( mean_face, (1, train_size))

A = train_data - M

L = np.dot( A.transpose(), A)
W, V = LA.eig(L)
print(W)
print(V)
U = np.dot(A, V)
# print(U.shape)

omega = np.dot( U.transpose(), A)


##############################################################################
# Eigenfaces: Recognition
print("Eigenfaces: Recognition")
dist_threshold = 1

file_test = ["01.centerlight",
				"01.happy",
				"01.normal",
				"02.normal",
				"03.normal",
				"07.centerlight",
				"07.happy",
				"07.normal",
				"10.normal",
				"11.centerlight",
				"11.happy",
				"11.normal",
				"12.normal",
				"14.happy",
				"14.normal",
				"14.sad",
				"15.normal",
				"apple1_gray"]

# Euclidean distance
def dist(omega_a, omega_b):
	row_a, col_a = omega_a.shape
	row_b, col_b = omega_b.shape
	if col_a != col_b:
		print("Waring: invalid data in dist().")
	if row_a != row_b:
		print("Waring: invalid data in dist().")
	distance = 0
	for i in range(0, row_a):
		for j in range(0, col_a):
			distance += (omega_a[i][j] - omega_b[i][j])*(omega_a[i][j] - omega_b[i][j])
	return math.sqrt(distance)


test_size = len(file_test)
for i in range(0, test_size):
	test_name = file_path + file_test[i] + ".jpg"
	test = imageio.imread(test_name)
	I = ( test.reshape((row*col,1)) - mean_face )
	omega_I = np.dot(U.transpose(), I)
	I_R = np.dot(U, omega_I)

	d0 = dist(I_R, I)
	print("d0 = %d" % (d0))
	# print(i)
	# print(I_R.shape)
	# plt_face(I_R)
	# Computer dist(omega_I, omega[:,i])
	distance = []
	for j in range(0, train_size):
		dist_i_j = dist(omega_I, omega[:,j].reshape((omega_I.shape)))
		distance.append(dist_i_j)
		# if dist_i_j < 1:
			# print("i is classified as: %d", j)
	print(distance)




#Display Image
# plt.imshow(test_image.reshape((row, col)), cmap = 'gray')
# plt.show()
# plt.savefig("grayImage_smoothed_with_Gaussian_filter.png")
# plt.close()

#
