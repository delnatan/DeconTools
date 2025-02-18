
# About this repo

# Design

## Deconvolution
For deconvolution, I want to do what Walter Carrington did in 1995 where the reconstructed image can be defined at an arbitrary pixel size (smaller than data). To do this, I first need to implement the forward and adjoint model for doing transformations between object-to-data space (forward) and data-to-object space.

Fluorescence intensities at the boundary is assumed to have contributions from object outside of the camera. To describe this process the forward model is:

blur -> downsample -> crop

The blurring and downsampling simulates what the objective lens does, and cropping simulates how light hits the detector (finite field of view). Practically this is done via FFTs.

O -> FT -> (element-wise multiply with OTF) -> (crop Fourier coefficients) -> inverse Ft -> unpad/crop in real-space X.

The adjoint operator does this in reverse:

pad -> upsample -> blur

X -> pad with zeros -> FT -> (pad Fourier coefficients) -> (element-wise multiply with OTF) -> inverse FT -> O


# TODO

- implement Ikoma et al. ADMM decon
- implement PSF averaging for phase retrieval (simplex algorithm to align)
- implement phase retrieval (ER method or Gerchberg-Saxton)
- implement SCAD prior/penalty operators (smoothly-clipped absolute deviation)


Notes:

The self-adjoint, D^t * D, of a Fourier cropping operator D, is a unit rectangle where its value is 1 within the cropping range and 0 otherwise. The adjoint operator is a zero-padding in Fourier-space.
