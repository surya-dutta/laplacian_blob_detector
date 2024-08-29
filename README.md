"# laplacian_blob_detector"

Implemented a Laplacian Blob Detector using Python and OpenCV, which is used for generating features that are invariant to scaling for feature tracking applications. Generated a Laplacian Scale Space using a Laplacian of Gaussian filter, and computed the maxima of the squared Laplacian response in the scale space. Performed Harris' Non-Max Suppression in the scale space in order to ignore the redundant blobs, and plotted the blobs at their characteristic scales on the image.
