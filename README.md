## This is a collection of small projects relating to signal processing and machine learning. ##

### adaptative_filter.py ###

This implements the LMS adn NLMS algorithms https://en.wikipedia.org/wiki/Least_mean_squares_filter . The idea is that an input signal (a sine wave is used as an exemple) is distorted and subjected to white gaussian noise. The adaptative filter tries to recover the original signal from he distoted noisy input.
The original input is given by the numpy array "x".The disitorted noisy signal is given by the numpy array "d".

The program plots the time convergence of the filter weights.

### image_processing.py ###

The image_processing file has 3 functions doing famous image processing tasks from scratch. The first one is resizing an image, done using interpolation.
The second one the a technique called "Histogram Equalization" https://en.wikipedia.org/wiki/Histogram_equalization, used for enchancing an image by making the darker areas dark and brighter areas white and making the image have uniform amounts of dark bright areas.
The third is a technique called "spread spectrum watermarking" https://www.researchgate.net/publication/228897428_Digital_image_watermarking_by_spread_spectrum. It hides information in the frequency domain representation of the image.

### raytracing.py ###

Very simple implematation of raytracing, The program outputs an image of a sphere.

### sindy.py ###

This is an implementation of the SINDY algorithm https://en.wikipedia.org/wiki/Sparse_identification_of_non-linear_dynamics, it recieves measured data as input and outputs a mathematical model explaining how the system behaves.

### toroidal_isomorphism.py ###

A torus is a donut shaped object, it has the property that going in any direction makes you be where you started. This program explores how linear functions behave on a torus. 
