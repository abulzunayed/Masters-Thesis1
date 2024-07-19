
# Implement Deep Neural network on DeepCEST
What is CEST MRI technique?

Chemical exchange saturation transfer (CEST) is a novel MR technique that enables imaging certain compounds at concentrations that are too low to impact the contrast of standard MR imaging and too low to directly be detected in MRS at typical water imaging resolution.
 
INTRODUCTION:
In this work, I learn how to achieve predict Lorentzian parameters 5-pool CEST MRI spectra with
corresponding uncertainty maps from uncorrected raw MRI Z-spectra using Deep neural networks as well
as reconstruct the prediction image.
As per research paper, the input data for a neural feed-forward network consisted of 7 T in vivo uncorrected
Z-spectra of a single B1 level, and a B1 map. The 7 T raw data were acquired using a 3D snapshot gradient
echo multiple interleaved mode saturation CEST sequence. These inputs were mapped voxel-wise to target
data consisting of Lorentzian amplitudes generated conventionally by 5-pool Lorentzian fitting of
normalized, denoised, B0- and B1-corrected Z-spectra. The deepCEST network was trained with Mean
square Error or Gaussian negative log-likelihood loss, providing an uncertainty quantification in addition
to the Lorentzian amplitudes Z-spectra of a single B1 level, and a B1 map.

METHODS
A Deep neural network used for processing pipeline to perform reconstruction of CEST contrasts image.
The conventional pipeline with the proposed deepCEST 7 T scheme uses in vivo Z-spectra and evaluated by
conventional methods using the evaluation pipeline described in Figure 1. Also, it was shown that neural
networks (NN) can be used and are effective to automate and accelerate the reconstruction from an
uncorrected CEST spectrum, forming the deepCEST approach. This deep NN is included into
the online image reconstruction of the scanner system to predict the CEST contrasts in ∼30 s with
uncertainty quantification to indicate the trustworthiness of the predictions.

![300170977-46539848-4109-4dd0-aa92-95e72ae75005](https://github.com/user-attachments/assets/9e846acf-a5a4-4e87-a45c-a691a4b160b5)


![341586274-4ff35a36-a2ea-4c0b-a54e-439b62711b75](https://github.com/user-attachments/assets/1d43a933-064f-43bb-8f67-49297d64bc1c)



![341586563-94f4678b-f167-4a4a-a8e0-f82cad071abd](https://github.com/user-attachments/assets/8ad3f47d-bc16-421d-bb23-085d24380522)


Achievable skills
By implementing this Deep neural Network for DeepCest, I have gathered the following knowledge:
▪ How to remove NAN values using mask- wise reference indices instead of directly deleting
elements.
▪ Detect each of image pixels/voxels values on plan data set in python.
▪ Visualized the image data on a single plan and manipulate dimension index-wise.
▪ Plot Histogram
▪ Remove crazy outliers without deleting matrix dimension/shape.
▪ Different methods of removing outliers e.g. z-score , I.Q.R , Percentile .
▪ Create Neural Network Model
▪ Use and effect of different loss functions (MSE, RMS, GNLL) and optimizers( Adam, GD, SGD).
▪ Reconstruct prediction image as per target image indices -wise elements.
▪ Testing and validation (r2-score, mse_error)

Research work:
Research article Link: https://doi.org/10.1002/mrm.29520
Data Source: https://github.com/cest-sources/deepCEST
Supervisor: Prof. Moritz Zaiss
Create by: Abul Kasem Mohammad Zunayed
Data: 11/05/2023
