#! /usr/bin/env python
__author__ 	= "Andrew Barthel"
__email__	= "abarthe1@asu.edu"

###############################################################
###############################################################
###############################################################
#
# Andrew Barthel - 1217975070
# abarthe1@asu.edu
# Arizona State University
# CSE 575 - Statistical Machine Learning
# Fall 2019
#
# Project 1 - Density Estimation and Classification
# Due - 9/10/19
#
# MLE Density Estimation and Naive Bayes Classification
#
###############################################################
###############################################################
###############################################################





###############################################################
####### PROGRAM SETUP - IMPORTS AND GLOBAL VARIABLES ##########
###############################################################

# Import statements for libraries needed
import scipy.io
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# Global variable for TA(s) and or Professor to print 
# output for testing purposes. This should be set to 
# zero for no output(default) and 1 for output. CLI.
testing = 0

###############################################################
###############################################################
###############################################################





###############################################################
################## FEATURE EXTRACTION #########################
###############################################################

# The method to extract the brightness of each image
# This is done by taking each image as a 28x28 matrix,
# and finding the mean value of the pixels over this
# matrix.
# Brightness = sum of all pixel values / number of pixels
#		- inputs: images = list of matricies of pixels for
#		all images in dataset.
#		- outputs: A vector of the average brightness over
#		all pixels for each image in dataset
def extract_brightness(images):

	# Define variables for middle computations.
	pixel_total = 0
	brightness_array = []

	# Loop through images and calculate the average pixel
	# value for each image.
	for x in images:
		for i in range(0, 28):
			for j in range(0, 28):
				pixel_total += x[i][j]

		# Store avg pixel value and reset variable for
		# next iteration.
		brightness_array.append(pixel_total / 784)
		pixel_total = 0

	# Return the vector for the average pixel brightness
	# for each image.
	return brightness_array

# This is the method to extract avg row variances.
# This is done by taking each image as a 28-rowed 
# matrix and taking the variance of each row. Then
# calculating the average of those variances.
# Avg(row-variance) = avg of all rows for : 
# (sum of (x_i - mean)^2)/number of columns
#		- inputs: images = list of matricies of 
#		pixels for all images in dataset.
#		- outputs: a vector of the avg row variance
#		of all images in dataset.
def extract_variance(images):

	# Define variables to keep track of middle computations.
	image_variance = []
	image_numerator = 0
	image_total = 0
	image_mean = 0
	row_variance = []
	row_pixels = []
	row_total = 0
	row_mean = 0
	var_numerator = 0

	# Loop through images and calculate variance for each row.
	for x in images:
		for i in range(0, 28):
			for j in range(0, 28):
				row_total += x[i][j]
				row_pixels.append(x[i][j])
			row_mean = row_total / 28
			for k in range(0, 28):
				var_numerator += (row_pixels[k] - row_mean)**2
			row_variance.append(var_numerator / 28)

			# Reset variables for next iteration.
			row_total = 0
			row_pixels = []
			row_mean = 0
			var_numerator = 0

		# Using the vector of row variances for image x,
		# calculate the average and store in a vector to return.
		total = 0
		for i in range(0, 28):
			total += row_variance[i]
		image_variance.append(total / 28)

		# Reset variables for next iteration.
		row_variance = []

	# Return the vector of all avg row variances for all images.
	return image_variance 

###############################################################
###############################################################
###############################################################





###############################################################
############### NAIVE BAYES CLASSIFIER ########################
###############################################################

# This method classifies testing images.
# This is done by taking all data(training and test)
# and passing the training into a learning algorithm 
# to create the probability density.  Then takes the test
# data and computes the probability for the new image
# and classifies it.  This function also calls a method
# to calculate accuracy.  It then returns this accuracy.
# Classification:
#			IF: probability of x being class 0 > probability
# of x being class 1. Then classify x as 0.
#			ELSE: Classify x as 1.
#		- inputs: all training and testing datasets 
#		and testing mode flag
#		- outputs: The calculated accuracies.
def naive_bayes(train_0_data, train_1_data, test_0_data, test_1_data, testing):

	# Define variables for middle computations.
	prob_0 = 0.0
	prob_1 = 0.0
	zero_labels = []
	one_labels = []

	# Print the estimated parameters for both classes if testing is set.
	if (testing == 1):
		mu1, mu2 = mu(train_0_data)
		sigma1, sigma2 = sigma(train_0_data, mu1, mu2)
		print("The estimated variables for class zero:")
		print("mu values of variable x1 and x2: %d and %d" % (mu1, mu2))
		print("sigma values of variable x1 and x2: %d and %d" % (sigma1, sigma2))
		mu1, mu2 = mu(train_1_data)
		sigma1, sigma2 = sigma(train_1_data, mu1, mu2)
		print("The estimated variables for class one:")
		print("mu values of variable x1 and x2: %d and %d" % (mu1, mu2))
		print("sigma values of variable x1 and x2: %d and %d" % (sigma1, sigma2))

	# For testing set zero, classify.
	for x in test_0_data:
		prob_0 = prob_zero(train_0_data, x)
		prob_1 = prob_one(train_1_data, x)

		# Classify and add to training set for dynamic learning.
		if prob_1 > prob_0:
			zero_labels.append(1)
			train_1_data.append(x)
		else:
			zero_labels.append(0)
			train_0_data.append(x)

	# For testing set one, classify.
	for x in test_1_data:
		prob_0 = prob_zero(train_0_data, x)
		prob_1 = prob_one(train_1_data, x)

		# Classify and add to training set for dynamic learning.
		if prob_0 > prob_1:
			one_labels.append(0)
			train_0_data.append(x)
		else:
			one_labels.append(1)
			train_1_data.append(x)

	# Print out the label lists if testing mode is set.
	if (testing == 1):
		print("The classification labels for testing set zero:")
		print(zero_labels)
		print("The classification labels for testing set one:")
		print(one_labels)

	# Calculate the accuracy and return it.
	return class_accuracy(zero_labels, one_labels)


# This method computes the probability for label 0.
# This is done by computing the probability for each
# feature of an image for class zero.
# p(x|y=0) = p(y=0)p(x_1|y=0)p(x_2|y=0) where p(0) = 0.5
#		- inputs: training data set zero, testing image
#		- outputs: probability for p(x|y=0)
def prob_zero(train_0_data, image):

	# Find the mu values and sigma values
	x1_mu, x2_mu = mu(train_0_data)
	x1_sigma, x2_sigma = sigma(train_0_data, x1_mu, x2_mu)

	# Find p(x_1|0) and p(x_2|0)
	prob0_x1 = prob_density_function(x1_mu, x1_sigma, image[0])
	prob0_x2 = prob_density_function(x2_mu, x2_sigma, image[1])

	# Find total probability and return
	return 0.5 * prob0_x1 * prob0_x2

# Thus method computes the probability for label 1.
# This is done by computing the probability for each
# feature of an image for class one.
# p(x|y=1) = p(y=1)p(x_1|y=1)p(x_2|y=1) where p(1) = 0.5
#		- inputs: training data set one, testing image
#		- outputs: probability for p(x|y=1)
def prob_one(train_1_data, image):

	# Find the mu values and sigma values
	x1_mu, x2_mu = mu(train_1_data)
	x1_sigma, x2_sigma = sigma(train_1_data, x1_mu, x2_mu)

	# Find p(x_1|1) and p(x_2|1)
	prob1_x1 = prob_density_function(x1_mu, x1_sigma, image[0])
	prob1_x2 = prob_density_function(x2_mu, x2_sigma, image[1])

	# Find total probability and return.
	return 0.5 * prob1_x1 * prob1_x2

###############################################################
###############################################################
###############################################################





###############################################################
###### MAXIMUM LIKELIHOOD ESTIMATION UTILITY METHODS ##########
###############################################################

# This method computes mu for both variables.
# This is done by finding the sample mean for
# both variables.
# mu = total value of x / number of x's
# 		- input: data = data set of image features
#		- output: mu for x1 and mu for x2
def mu(data):

	# Define variables for middle computations.
	tot1 = 0
	tot2 = 0

	# Find totals for both x1 and x2.
	for x in data:
		tot1 += x[0]
		tot2 += x[1]
	
	# Find the two mu values and return them.
	return tot1/len(data), tot2/len(data)

# This method computes sigma for both variables.
# This is done by finding the sample sigma^2 for
# both variables.
# sigma = sum of (xi - mu)^2 / number of xi's
#				for all x1 and x2
#		- input: data = data set of image features
#		- output: sigma for x1 and sigma for x2
def sigma(data, mu1, mu2):

	# Define variables for middle computations.
	num1 = 0
	num2 = 0

	# Find numerator for sigma formula.
	for x in data:
		num1 += (x[0] - mu1)**2
		num2 += (x[1] - mu2)**2
	
	# Find the two sigmas and return them.
	return num1/len(data), num2/len(data)

# The Probability Density Function.
# This function defines the probability density 
# for the given distribution. Since this is a 
# normal dictribution:
# p(x) = (1/(2pisigma^2))exp^(-(x-mu)^2/(2sigma^2))
# 		- inputs: mu = mu for distribution.
#				  sigma = sigma for distribution.
#				  x = random variable as input to PDF.
#		- output: The probability of variable x.
def prob_density_function(mu, sigma, x):

	# Find the power that e is raised to and its constant
	exp_power = ((x - mu)**2)/(2 * sigma)
	exp_const = 1 / (math.sqrt(2 * math.pi * sigma))

	# Solve for the probability function and return
	probf = exp_const * math.exp(-exp_power)
	return probf

###############################################################
###############################################################
###############################################################





###############################################################
############ CLASSIFICATION ACCURACY REPORT ###################
###############################################################

# The accuracy the naive bayes classifier ran on.
# This computes the accuracy for testing set zero,
# testing set one, and overall testing set.
#		- inputs: zero_labels = dataset labels for
#		set zero.
#				  one_labels = dataset labels for set one.
#		- outputs: set zero accuracy, set one accuracy, total accuracy
def class_accuracy(zero_labels, one_labels):

	# Get accuracy of the zero testing set.
	total0 = 0.0
	correct0 = 0.0
	for x in zero_labels:
		if (x == 0):
			correct0 += 1
		total0 += 1
	accuracy_zero = correct0/total0 

	# Get accuracy of the one testing set.
	total1 = 0.0
	correct1 = 0.0
	for x in one_labels:
		if (x == 1):
			correct1 += 1
		total1 += 1
	accuracy_one = correct1/total1

	# Get total accuracy and return
	tot = (correct0+correct1)/(total0+total1)
	return (accuracy_zero, accuracy_one, tot)

###############################################################
###############################################################
###############################################################





###############################################################
######################## MAIN #################################
###############################################################

# Main function of project. Handles args, opens datasets,
# parses images, calls to extract features, formats data,
# testing plot (optional), then calls the classifier.
# 		- inputs: void
#		- returns: sys exit with success code zero
def main():

	# Handle cli args for testing purposes.
	# No error handling - beware.
	# expects format:
	# 		1.) "python main.py" (defaults to 0)
	# 		2.) "python main.py --testing {0,1}"
	if (len(sys.argv) == 3) and (sys.argv[1] == "--testing"):
		testing = int(sys.argv[2])
	else:
		testing = 0

	# Open the datasets.
	test_0_img = scipy.io.loadmat('test_0_img.mat')
	test_1_img = scipy.io.loadmat('test_1_img.mat')
	train_0_img = scipy.io.loadmat('train_0_img.mat')
	train_1_img = scipy.io.loadmat('train_1_img.mat')

	# Parse the image files for the pixels.
	test_0_img_data = test_0_img["target_img"].T
	test_1_img_data = test_1_img["target_img"].T
	train_0_img_data = train_0_img["target_img"].T
	train_1_img_data = train_1_img["target_img"].T

	# Extract the average brightness for each image.
	test_0_bright = extract_brightness(test_0_img_data)
	test_1_bright = extract_brightness(test_1_img_data)
	train_0_bright = extract_brightness(train_0_img_data)
	train_1_bright = extract_brightness(train_1_img_data)

	# Extract the average variance for each row in each picture.
	test_0_var = extract_variance(test_0_img_data)
	test_1_var = extract_variance(test_1_img_data)
	train_0_var = extract_variance(train_0_img_data)
	train_1_var = extract_variance(train_1_img_data)

	# Create 2-d vectors of data
	test_0_data = []
	for i in range(0, len(test_0_bright)):
		test_0_data.append((test_0_bright[i], test_0_var[i]))
	test_1_data = []
	for i in range(0, len(test_1_bright)):
		test_1_data.append((test_1_bright[i], test_1_var[i]))
	train_0_data = []
	for i in range(0, len(train_0_bright)):
		train_0_data.append((train_0_bright[i], train_0_var[i]))
	train_1_data = []
	for i in range(0, len(train_1_bright)):
		train_1_data.append((train_1_bright[i], train_1_var[i]))

	# Print and plot feature vectors if set to testing mode.
	if (testing == 1):

		# Print Training and Testing feature vectors.
		print("The feature vectors for training set of zero:")
		print(train_0_data)
		print("The feature vectors for training set of one:")
		print(train_1_data)
		print("The feature vectors for testing set of zero:")
		print(test_0_data)
		print("The feature vectors for testing set of one:")
		print(test_1_data)

		# Plot Training and Testing data.
		plt.figure(1)
		plt.subplot(211)
		plt.plot(train_0_bright, train_0_var, 'o')
		plt.plot(train_1_bright, train_1_var, '1')
		plt.ylabel("Variance")
		plt.xlabel("Brightness")
		plt.subplot(212)
		plt.plot(test_0_bright, test_0_var, 'o')
		plt.plot(test_1_bright, test_1_var, '1')
		plt.ylabel("Variance")
		plt.xlabel("Brightness")
		plt.suptitle("Training Data then Testing Data")
		plt.show()

	# Use Naive Bayes to Classify new data and find accuracies.
	accuracies = naive_bayes(train_0_data, train_1_data, test_0_data, test_1_data, testing)

	# Print accuracies - with or w/o testing mode set.
	print("The accuracy of the Naive Bayes Classifier on testing set zero: %5.2f%%" % (accuracies[0] * 100))
	print("The accuracy of the Naive Bayes Classifier on testing set one: %5.2f%%" % (accuracies[1] * 100))
	print("The accuracy of the Naive Bayes Classifier on the testing set overall: %5.2f%%" % (accuracies[2] * 100))


	# Successfully terminate program
	sys.exit(0)
	
# Ensure to call main function
if __name__ == '__main__':
	main()
else:
	print('Something is really wrong....There is no main function, man?')
	sys.exit(1)

###############################################################
###############################################################
###############################################################