/*Dev Vaibhav
Spring 2023 CS 5330
Project 4: Calibration and Augmented Reality
*/

// Calculate camera calibration parameters
int calibrate_camera();

// Project the virtual object on the image
int project(int &source, char *target_filename_char_star, int &alter_base);

// Superimpose an image/ video on the original frame if 4 Aruco markers are detected
int arucoTV(int &ar_source, char *target_filename_char_star);
