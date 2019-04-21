# Principal-Component-Analysis-to-generate-good-features-for-Handwritten-Digit-Recognition

We explore how to use PCA to generate “good” features for Handwritten Digit Recognition using the
USPS data set. The dataset has been reduced and pre-processed. The files usps.train, usps.valid and usps.test
are in CSV format, with the first column being the digit label and the next 256 columns representing gray scale intensities of the 16 × 16 digit.

The features are all in the range [0, 1]. X100 corresponds to using all the features to capture 100% of the variance, and
k100 = 256 denote the total dimension of the original data set.
