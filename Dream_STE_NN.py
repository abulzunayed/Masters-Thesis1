# %% S0. SETUP env
import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import torch
from matplotlib import pyplot as plt
import util

# makes the ex folder your working directory
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # had to add this line so the spyder console doesn't crash when running the sequence
os.chdir(os.path.abspath(os.path.dirname(__file__)))

## imports for image reconstruction
from reconstruction import adaptive_combine
from skimage.restoration import unwrap_phase

## imports for WASABI
import scipy
from scipy.interpolate import RegularGridInterpolator

## import for fitting
from scipy.stats import linregress

## import for masking
import matplotlib.image

## import for alternative phantom
import random

experiment_id = 'DREAM_STE_1_3'

# %% S1. SETUP sys
'''
## default scanner limits
system = Opts(
    max_grad=28, 
    grad_unit='mT/m', 
    max_slew=150, 
    slew_unit='T/m/s', 
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6, 
    adc_dead_time=20e-6,
    grad_raster_time=50*10e-6)
'''
#limits die Moritz mir gesagt hat
system = pp.Opts(
    max_grad=80, 
    grad_unit='mT/m', 
    max_slew=200, 
    slew_unit='T/m/s', 
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6, 
    adc_dead_time=20e-6,
    grad_raster_time=10e-6 # actually 50e-6 but had to change it so extended gradients work
)

# %% S2. DEFINE the sequence 
seq = pp.Sequence()

# Define FOV and resolution
fov = 220e-3
slice_thickness=8e-3
sz = (64, 64)  # spin system size / resolution
Nread = 64  # frequency encoding steps/samples
Nphase = 64  # phase encoding steps/samples
zoom = 1
ncoils = 20 #16 for: links, Mitte, Halbrechts, Rechts, Mitte_v1, DrehungHalblinks, DrehungLinks, DrehungHalbRechts, DrehungRechts  #20 for: Halblinks
t0 = 5e-4 # smallest time interval [s] used in all events
#t1 = t0#*5 # t0 has to be converted for the extended gradients
gx_read_amp = (2*Nread*zoom)/(fov*4*t0) # amplitude for G_m (gx_read) part of gx_ext
gx_pre_amp = -7*gx_read_amp/2 # amplitude for G_m1 (gx_pre) part of gx_ext
# -> maybe change the timing system, so that t0 isn't used for everything and events can be timed better individually


# Define rf events
# STEAM rf pulses:
rf1 = pp.make_block_pulse(flip_angle=55 * np.pi / 180, duration=t0, slice_thickness=slice_thickness, system=system)
rf2 = pp.make_block_pulse(flip_angle=55 * np.pi / 180, phase_offset=180*np.pi/180, duration=t0, slice_thickness=slice_thickness, system=system)

rf1.delay=0
rf2.delay=0

# FLASH readout pulse and slice selction gradients:
rf3, gz3, gzr3 = pp.make_sinc_pulse(
    flip_angle=15 * np.pi/180,
    duration=2*t0,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system,
    return_gz=True
)

# Define other gradients and ADC events
gx_ext = pp.make_extended_trapezoid(channel='x', amplitudes=np.array([0,gx_pre_amp,0,gx_read_amp,gx_read_amp,10*gx_read_amp,0]), times=np.array([0,1*t0,2*t0,3*t0,7*t0,8*t0,9*t0]), system=system)
gx_m2 = pp.make_trapezoid(channel='x', area=-gx_read_amp*2*t0, duration=11*t0+2*gz3.fall_time-system.rf_ringdown_time, system=system)
gx_spoil = pp.make_trapezoid(channel='x', area=6*gx_read_amp*t0, duration=1e-3, system=system)
adc = pp.make_adc(num_samples=Nread*2, duration=4*t0, delay=3*t0, phase_offset=0*np.pi/180,system=system)

#dummies
dummies = 3

#rf spoiling
rf_phase = 0
rf_inc = 0
rf_spoiling_inc=84

#centric reordering
phase_enc__gradmoms = torch.arange(0,Nphase,1)-Nphase//2

permvec=np.zeros((Nphase,),dtype=int)
permvec[0]=0
for i in range(1,int(Nphase//2+1)):
    permvec[i*2-1]=-i
    if i <Nphase/2:
        permvec[i*2]=i
permvec+=Nphase//2
phase_enc__gradmoms=phase_enc__gradmoms[permvec]

# ======
# CONSTRUCT SEQUENCE
# ======

'''
# MP:
rf_prep, _= make_block_pulse(flip_angle=180 * math.pi / 180, duration=1e-3, system=system)
#FLAIR
seq.add_block(rf_prep)
seq.add_block(make_delay(2.7))
seq.add_block(gx_spoil)
'''
'''
# DIR
seq.add_block(rf_prep)
seq.add_block(make_delay(0.45))
seq.add_block(gx_spoil)
'''

#seq.add_block(pp.make_delay(10)) # delay to ensure total relaxation of magnetization when measuring multiple sequences consecutively

#STEAM block
seq.add_block(rf1)
seq.add_block(gx_m2)
seq.add_block(rf2)
seq.add_block(gx_spoil)

#dummy block
for i in range(0, dummies):
    #rf spoiling
    rf3.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC
    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]   # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]        # increment additional pahse
    #dummies
    seq.add_block(rf3,gz3)     
    seq.add_block(gx_ext,gzr3) 

#readout block
rf3.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC
rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]   # increase increment
rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]        # increment additional pahse

seq.add_block(rf3,gz3)
seq.add_block(gx_ext,gzr3,adc)

for ii in range(1, Nphase):
    #rf spoiling
    rf3.phase_offset = rf_phase / 180.0 * np.pi   # set current rf phase
    adc.phase_offset = rf_phase / 180.0 * np.pi  # follow with ADC
    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]   # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]        # increment additional pahse
    
    gy_amp = (2*phase_enc__gradmoms[ii]*zoom)/(fov*2*t0) # set amplitude for y-gradients
    gy_ext = pp.make_extended_trapezoid(channel='y', amplitudes=np.array([0,gy_amp,0,0,-gy_amp,0]), times=np.array([0,1*t0,2*t0,7*t0,8*t0,9*t0]), system=system)
    seq.add_block(rf3,gz3)
    seq.add_block(gx_ext,gy_ext,gzr3,adc)
    
seq.add_block(pp.make_delay(4)) # delay to ensure total relaxation of magnetization when measuring multiple sequences consecutively

# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc, t_adc = util.pulseq_plot(seq, clear=False, figid=(11,12))
#   
if 0:
    sp_adc,t_adc =seq.plot(clear=True)


# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
'''
seq.write('out/' + experiment_id +'.seq')
'''
# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
'''
sz = [64, 64]

if 1:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)
    # Manipulate loaded data
    obj_p.T2dash[:] = 30e-3
    obj_p.D *= 0 
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
    # Store PD for comparison
    PD = obj_p.PD
    B0 = obj_p.B0
    B1 = obj_p.B1
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = mr0.CustomVoxelPhantom(
        pos=[[-0.4, -0.4, 0], [-0.4, -0.2, 0], [-0.3, -0.2, 0], [-0.2, -0.2, 0], [-0.1, -0.2, 0]],
        PD=[1.0, 1.0, 0.5, 0.5, 0.5],
        T1=1.0,
        T2=0.1,
        T2dash=0.1,
        D=0.0,
        B0=0,
        voxel_size=0.1,
        voxel_shape="box"
    )
    # Store PD for comparison
    PD = obj_p.generate_PD_map()
    B0 = torch.zeros_like(PD)

obj_p.plot()
# Convert Phantom into simulation data
obj_p = obj_p.build()

# change size and orientation of phantom data for later comparison
PD.resize_(64,64)
B0.resize_(64,64)
B1.resize_(64,64)

PD=np.flip(np.rot90(PD,3),1)
B0=np.flip(np.rot90(B0,3),1)
B1=np.flip(np.rot90(B1,3),1)
'''

# alternative simulation phantoms

sz = [64, 64]
subject_list = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44,
45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
subject_num = 54 #random.choice(subject_list) # random subject from subject_list, alternativly select one manually
phantom_path = f'../data/brainweb/output/subject{subject_num:02d}.npz'
slice_num = 216 #center slice
if 1:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.brainweb(phantom_path).slices([slice_num]) #original resolution 432x432x432
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)
    # Manipulate loaded data
    obj_p.T2dash[:] = 30e-3
    obj_p.D *= 0 
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
    # Store PD for comparison
    PD = obj_p.PD
    B0 = obj_p.B0
    B1 = obj_p.B1
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = mr0.CustomVoxelPhantom(
        pos=[[-0.4, -0.4, 0], [-0.4, -0.2, 0], [-0.3, -0.2, 0], [-0.2, -0.2, 0], [-0.1, -0.2, 0]],
        PD=[1.0, 1.0, 0.5, 0.5, 0.5],
        T1=1.0,
        T2=0.1,
        T2dash=0.1,
        D=0.0,
        B0=0,
        voxel_size=0.1,
        voxel_shape="box"
    )
    # Store PD for comparison
    PD = obj_p.generate_PD_map()
    B0 = torch.zeros_like(PD)

obj_p.plot()
# Convert Phantom into simulation data
obj_p = obj_p.build()

def resample(tensor: torch.Tensor) -> torch.Tensor:
    # Introduce additional dimensions: mini-batch and channels
    return torch.nn.functional.interpolate(
        tensor[None, None, ...], size=(sz[0], sz[1], 1), mode='area'
    )[0, 0, ...]    

with np.load(phantom_path) as data:
    P_WM = torch.tensor(data['tissue_WM'])[:,:,slice_num]
    P_GM = torch.tensor(data['tissue_GM'])[:,:,slice_num]
    P_CSF = torch.tensor(data['tissue_CSF'])[:,:,slice_num]

P_WM = resample(P_WM[:,:,None])
P_GM = resample(P_GM[:,:,None])
P_CSF = resample(P_CSF[:,:,None])

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("Probability Map, white matter")
plt.imshow(P_WM[:, :, 0].T.cpu(), vmin=0, origin="lower")
plt.colorbar()
plt.subplot(132)
plt.title("Probability Map, grey matter")
plt.imshow(P_GM[:, :, 0].T.cpu(), vmin=0, origin="lower")
plt.colorbar()
plt.subplot(133)
plt.title("Probability Map, CSF")
plt.imshow(P_CSF[:, :, 0].T.cpu(), vmin=0, origin="lower")
plt.colorbar()

# change size and orientation of phantom data for later comparison
PD.resize_(64,64)
B0.resize_(64,64)
B1.resize_(64,64)

PD=np.flip(np.rot90(PD,3),1)
B0=np.flip(np.rot90(B0,3),1)
B1=np.flip(np.rot90(B1,3),1)



# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot
use_simulation = True    # for Simulation
#use_simulation = False     # for vivo,  Created zunayed

if use_simulation:
    seq_file = mr0.PulseqFile("out/external.seq")
    seq0 = mr0.Sequence.from_seq_file(seq_file)
    seq0.plot_kspace_trajectory()
    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0, obj_p)
    
    plt.close(11);plt.close(12)
    sp_adc, t_adc = util.pulseq_plot(seq, clear=False, signal=signal.numpy())
else:
    signal = util.get_signal_from_real_system('E:/Dream_MR0_core/data/invivo_230620/' + 'DREAM_STE_1_3' + '.seq.dat', Nphase, Nread*2)
    
    sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
    sp_adc.plot(t_adc,np.abs(signal.numpy()))
    
    
# PLOT sequence with signal in the ADC subplot
#plt.close(11);plt.close(12)
#sp_adc, t_adc = util.pulseq_plot(seq, clear=False, signal=signal.numpy()) # doesn't work for invivo
#sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
#sp_adc.plot(t_adc,np.abs(signal.numpy()))
'''
# additional noise as simulation is perfect
z = 1e-5*np.random.randn(signal.shape[0], 2).view(np.complex128) 
signal+=z
'''
# %% S6: MR IMAGE RECON of signal ::: #####################################

if use_simulation:
    #simulation
    fig=plt.figure(); # fig.clf()
    plt.subplot(411); plt.title('ADC signal')
    
    plt.plot(torch.real(signal),label='real')
    plt.plot(torch.imag(signal),label='imag')
    
    major_ticks = np.arange(0, 2*Nphase*Nread, Nread*2) # this adds ticks at the correct position szread
    ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()
    
    spectrum=torch.reshape((signal),(Nphase,Nread*2)).clone().transpose(1,0)
    
    # centric reordering
    kspace_adc1=spectrum[0:Nread,:]
    kspace_adc2=spectrum[Nread:,:]
    ipermvec=np.arange(len(permvec))[np.argsort(permvec)]
    kspace1=kspace_adc1[:,ipermvec]
    kspace2=kspace_adc2[:,ipermvec]
    
    #recon of first kspace
    space1 = torch.zeros_like(kspace1)
    # fftshift
    kspace1_1=torch.fft.fftshift(kspace1,0); kspace1_1=torch.fft.fftshift(kspace1_1,1)
    #FFT
    space1 = torch.fft.ifft2(kspace1_1,dim=(0,1))
    # fftshift
    space1=torch.fft.ifftshift(space1,0); space1=torch.fft.ifftshift(space1,1)
    
    img_STE = space1
    
    #recon of second kspace
    space2 = torch.zeros_like(kspace2)
    # fftshift
    kspace2_1=torch.fft.fftshift(kspace2,0); kspace2_1=torch.fft.fftshift(kspace2_1,1)
    #FFT
    space2 = torch.fft.ifft2(kspace2_1,dim=(0,1))
    # fftshift
    space2=torch.fft.ifftshift(space2,0); space2=torch.fft.ifftshift(space2,1)
    
    img_FID = space2
    
    img_STE = np.flip(np.rot90(img_STE,3),1)
    
    img_FID = np.flip(np.rot90(img_FID,3),1)
    
    Dream_FID=np.abs(img_FID)         # created by zunayed
    Dream_STE=np.abs(img_STE)         # created by zunayed
    
    plt.subplot(345); plt.title('k-space_STE')
    plt.imshow(np.abs(kspace1))
    plt.subplot(349); plt.title('k-space_r_STE')
    plt.imshow(np.log(np.abs(kspace1)))

    plt.subplot(346); plt.title('k-space_FID')
    plt.imshow(np.abs(kspace2))
    plt.subplot(3,4,10); plt.title('k-space_r_FID')
    plt.imshow(np.log(np.abs(kspace2)))
    
    plt.subplot(347); plt.title('FFT-magnitude_STE', fontsize=15)
    plt.imshow(np.abs(img_STE),vmin=0,vmax=0.083,origin='lower'); plt.axis('off'); plt.colorbar()
    plt.subplot(3,4,11); plt.title('FFT-magnitude_FID', fontsize=15)
    plt.imshow(np.abs(img_FID),vmin=0,vmax=0.083,origin='lower'); plt.axis('off'); plt.colorbar()

    plt.subplot(348); plt.title('FFT-phase_STE', fontsize=15)
    plt.imshow(np.angle(img_STE),vmin=-np.pi,vmax=np.pi,origin='lower'); plt.axis('off'); plt.colorbar()
    plt.subplot(3,4,12); plt.title('FFT-phase_FID', fontsize=15)
    plt.imshow(np.angle(img_FID),vmin=-np.pi,vmax=np.pi,origin='lower'); plt.axis('off'); plt.colorbar()

else:
    #scanner
    fig=plt.figure(); # fig.clf()
    plt.subplot(411); plt.title('ADC signal')
    
    plt.plot(torch.real(signal[:,13]),label='real') # real and imaginary part of the signal from one channel
    plt.plot(torch.imag(signal[:,13]),label='imag')
    
    major_ticks = np.arange(0, 2*Nphase*Nread, Nread*2) # this adds ticks at the correct position szread
    ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()
    
    # pulled recon of first channel out of the for-loop, so that i can define space1_all and space2_all (probably possible to improve that)
    spectrum=torch.reshape((signal),(Nphase,Nread*2,ncoils)).clone().transpose(1,0)
    
    # centric reordering
    kspace_adc1=spectrum[0:Nread,:,0]
    kspace_adc2=spectrum[Nread:,:,0]
    ipermvec=np.arange(len(permvec))[np.argsort(permvec)]
    kspace1_0=kspace_adc1[:,ipermvec]
    kspace2_0=kspace_adc2[:,ipermvec]
    
    
    space1_0 = torch.zeros_like(kspace1_0)
    # fftshift
    spectrum=torch.fft.fftshift(kspace1_0);
    #FFT
    space1_0 = torch.fft.ifft2(spectrum)
    # fftshift
    space1_0=torch.fft.ifftshift(space1_0);
    #space=torch.sum(space.abs(),2)

    space2_0 = torch.zeros_like(kspace2_0)
    # fftshift
    spectrum=torch.fft.fftshift(kspace2_0);
    #FFT
    space2_0 = torch.fft.ifft2(spectrum)
    # fftshift
    space2_0=torch.fft.ifftshift(space2_0);
    
    '''
    # plot kspace and FFT of every single channel
    fig=plt.figure(10);
    
    plt.subplot(241); plt.title('k-space_STE')
    plt.imshow(np.abs(kspace1_0))
    plt.subplot(245); plt.title('k-space_r_STE')
    plt.imshow(np.log(np.abs(kspace1_0)))

    plt.subplot(242); plt.title('k-space_FID')
    plt.imshow(np.abs(kspace2_0))
    plt.subplot(2,4,6); plt.title('k-space_r_FID')
    plt.imshow(np.log(np.abs(kspace2_0)))

    plt.subplot(243); plt.title('FFT-magnitude_STE')
    plt.imshow(np.abs(space1_0)); plt.colorbar()
    plt.subplot(2,4,7); plt.title('FFT-magnitude_FID')
    plt.imshow(np.abs(space2_0)); plt.colorbar()

    plt.subplot(244); plt.title('FFT-phase_STE')
    plt.imshow(np.angle(space1_0),vmin=-np.pi,vmax=np.pi); plt.colorbar()
    plt.subplot(2,4,8); plt.title('FFT-phase_FID')
    plt.imshow(np.angle(space2_0),vmin=-np.pi,vmax=np.pi); plt.colorbar()
    '''
    
    space1_all = space1_0[None,:,:]
    space2_all = space2_0[None,:,:]
    
    #recon of the remaining 19 channels:
    for i in range(1,ncoils):
        spectrum=torch.reshape((signal),(Nphase,Nread*2,ncoils)).clone().transpose(1,0)
        
        # centric reordering
        kspace_adc1_i=spectrum[0:Nread,:,i]
        kspace_adc2_i=spectrum[Nread:,:,i]
        ipermvec=np.arange(len(permvec))[np.argsort(permvec)]
        kspace1_i=kspace_adc1_i[:,ipermvec]
        kspace2_i=kspace_adc2_i[:,ipermvec]
        
        space1_i = torch.zeros_like(kspace1_i)
        # fftshift
        spectrum=torch.fft.fftshift(kspace1_i);
        #FFT
        space1_i = torch.fft.ifft2(spectrum)
        # fftshift
        space1_i=torch.fft.ifftshift(space1_i);
        #space=torch.sum(space.abs(),2)

        space2_i = torch.zeros_like(kspace2_i)
        # fftshift
        spectrum=torch.fft.fftshift(kspace2_i);
        #FFT
        space2_i = torch.fft.ifft2(spectrum)
        # fftshift
        space2_i=torch.fft.ifftshift(space2_i);
        
        '''
        # plot kspace and FFT of every single channel
        fig=plt.figure(10+i);
        
        plt.subplot(241); plt.title('k-space_STE')
        plt.imshow(np.abs(kspace1_i))
        plt.subplot(245); plt.title('k-space_r_STE')
        plt.imshow(np.log(np.abs(kspace1_i)))

        plt.subplot(242); plt.title('k-space_FID')
        plt.imshow(np.abs(kspace2_i))
        plt.subplot(2,4,6); plt.title('k-space_r_FID')
        plt.imshow(np.log(np.abs(kspace2_i)))

        plt.subplot(243); plt.title('FFT-magnitude_STE')
        plt.imshow(np.abs(space1_i)); plt.colorbar()
        plt.subplot(2,4,7); plt.title('FFT-magnitude_FID')
        plt.imshow(np.abs(space2_i)); plt.colorbar()

        plt.subplot(244); plt.title('FFT-phase_STE')
        plt.imshow(np.angle(space1_i),vmin=-np.pi,vmax=np.pi); plt.colorbar()
        plt.subplot(2,4,8); plt.title('FFT-phase_FID')
        plt.imshow(np.angle(space2_i),vmin=-np.pi,vmax=np.pi); plt.colorbar()
        '''
        #putting the data from all 20 channels into one tensor
        space1_all = torch.cat((space1_all,space1_i[None,:,:]),0)
        space2_all = torch.cat((space2_all,space2_i[None,:,:]),0)
        
    #modify the dimension of the data tensor so that it is possible to use adaptive_combine
    space1_all = space1_all[:,:,:,None]
    space2_all = space2_all[:,:,:,None]
    
    #adaptive combine: combines data from all channels
    #adaptive combine of FID data:
    acom_FID, weights_FID = adaptive_combine(space2_all)
    #adaptive combine of STE data with weights of FID adaptive_combine:
    sz = space1_all.shape
    nc = sz[0] # coils first!
    n = torch.tensor(sz[1:])
    acom_STE = torch.sum(weights_FID * space1_all, dim=0).reshape((*n, ))
    
    #resizing resulting tensors
    img_STE = acom_STE.resize(64,64)
    img_FID = acom_FID.resize(64,64)
    
    #commands to rotate/flip resulting images to match reference images:
    img_STE = torch.rot90(img_STE,3)
    img_STE = torch.flip(img_STE,(0,))
    
    img_FID = torch.rot90(img_FID,3)
    img_FID = torch.flip(img_FID,(0,))
    
    # Save each sequence  FID and STE for future Neural network
    Dream_vivo_FID=np.abs(img_FID)         # created by zunayed
    Dream_vivo_STE=np.abs(img_STE)         # created by zunayed

    #plot magnitude and phase of STE and FID:
    fig=plt.figure(60)

    plt.subplot(221); plt.title('FFT-magnitude_STE', fontsize=15) # all channels
    plt.imshow(np.abs(img_STE), vmin=0, vmax=8.6e-6, origin='lower'); plt.axis('off'); plt.colorbar() # , origin='lower' for reversed y axis
    plt.subplot(223); plt.title('FFT-magnitude_FID', fontsize=15) # all channels
    plt.imshow(np.abs(img_FID), vmin=0, vmax=8.6e-6, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(222); plt.title('FFT-phase_STE [rad]', fontsize=15) # all channels 
    plt.imshow(np.angle(img_STE), vmin=-np.pi, vmax=np.pi, origin='lower'); plt.axis('off'); plt.colorbar()
    plt.subplot(224); plt.title('FFT-phase_FID [rad]', fontsize=15) # all channels
    plt.imshow(np.angle(img_FID), vmin=-np.pi, vmax=np.pi, origin='lower'); plt.axis('off'); plt.colorbar()

# %% S7: MASKING

# choose wether or not to use a mask
masking = 1
if masking:
    # choose whether or not to use a threshold mask (for simulation its always a threshold mask because it works well on the simulation phantom)
    thresh_mask = 1  # original it was 1
    
    # function for threshold masking
    def mask_im(input_array, threshold, mask_values):
        mask = np.ones_like(input_array)
        mask[input_array<threshold] = mask_values
        return mask
    
    if use_simulation==1:   #original it was 1
        #PDM=np.load('mask_array.npy')  # use saved PDM to stop generate different mask value from PD
        mask_zero= mask_im(PD, 0.7, 0) #PD, 0.01 for alternative phantoms # zunayed use PDM instead of PD
        mask_NaN = mask_im(PD, 0.7, np.NaN) #PD, 0.01 for alternative phantoms # zunayed use PDM instead of PD
        #mask_array=PD
        #np.save('mask_array.npy' , mask_array)  # save PD so that we can always use it in future
    else:
        if thresh_mask:
            # new mask for vivo
            mask_NaN=torch.load('manual_mask')
            mask_zero= np.ones_like(mask_NaN)
            mask_zero[mask_NaN != 1] = 0
            '''
            #invivo
            mask_zero = mask_im(np.abs(img_STE), 4e-7, 0) #5e-7
            mask_NaN = mask_im(np.abs(img_STE), 4e-7, np.NaN) #5e-7
            '''
            
        else:
            '''
            # variant 1:
            #1: export one image where you can see the ROI 
            #matplotlib.image.imsave('STE_mag.png',np.abs(img_STE).numpy()) #only necessary if you create the mask for the frist time
            #2: open it in paint, cut out the ROI, copy the remaining image to another paint file, save the file as a .png
            #3: import the cutout.png
            new_mask2=matplotlib.pyplot.imread('mask2.png')
            #4: cut out the ROI from the loaded data (have a look at the array with plt.imshow before)
            new_new_mask2=new_mask2[1:65,1:65,0]
            #5: create binary mask again
            mask_zero = mask_im(new_new_mask2,1,0)
            mask_NaN = mask_im(new_new_mask2,1,np.NaN)
            '''
            '''
            # variant 2:
            #1: export one image where you can see the ROI 
            #matplotlib.image.imsave('STE_mag.png',np.abs(img_STE).numpy()) #only necessary if you create the mask for the frist time
            #2: open it in paint, cut out the ROI, copy the ROI to another paint file, save the file as a .png
            #3: import the ROI.png
            new_mask3=matplotlib.pyplot.imread('mask3.png')
            #4: cut out the ROI from the loaded data (have a look at the array with plt.imshow before)
            new_new_mask3=new_mask3[1:65,1:65,1]
            new_new_mask3[new_new_mask3==1]=0 # change ones to zeros, so that mask_im works
            #5: create binary mask again
            mask_zero = mask_im(new_new_mask3,0.3,0)
            mask_NaN = mask_im(new_new_mask3,0.3,np.NaN)
            '''
            
            # variant 3: whiten single pixels on the ROI in paint
            new_mask5=matplotlib.pyplot.imread('mask6.png')# mask5 for Thesis #mask6 for newer 
            new_new_mask5=new_mask5[:,:,1] #1:65 for Thesis #:,: for newer
            new_new_mask5[new_new_mask5==1]=0
            mask_zero = mask_im(new_new_mask5,0.2,0) #0.3 for Thesis #0.2 for newer
            mask_NaN = mask_im(new_new_mask5,0.2,np.NaN)
else:
    mask_zero = np.ones_like(B1)
    mask_NaN = np.ones_like(B1)

# plot mask with zeros and NaNs (zero mask is used for further calculations, NaN mask is used when showing the images)
fig=plt.figure();
plt.subplot(121); plt.title('mask_zeros', fontsize=15)
plt.imshow(mask_zero, vmin=0, vmax=1, origin='lower'); plt.colorbar()

plt.subplot(122); plt.title('mask_NaN', fontsize=15)
plt.imshow(mask_NaN, vmin=0, vmax=1, origin='lower'); plt.colorbar()

#%%     marge all  FID and STE image   #################

# FID data
#train18_FID=np.reshape(Dream_FID*mask_zero, (4096,1))               # chnage 5
#np.save('train_FID.npy', train_FID)

#marge_FID= np.load('marge_FID.npy')
#marge_FID=np.concatenate((marge_FID, train18_FID), axis=1)            # chnage 18
#np.save('marge_FID.npy', marge_FID)

# STE data
#train18_STE=np.reshape(Dream_STE*mask_zero, (4096,1))               # chnage 18
#np.save('train_STE.npy', train_STE)

#marge_STE= np.load('marge_STE.npy')
#marge_STE=np.concatenate((marge_STE, train18_STE), axis=1)            # chnage 18
#np.save('marge_STE.npy', marge_STE)
        
#fig=plt.figure()
#plt.subplot(121); plt.title('FFT-magnitude_STE', fontsize=15)
#plt.imshow(Dream_STE*mask_zero,vmin=0,vmax=0.083,origin='lower'); plt.axis('off'); plt.colorbar()
#plt.subplot(122); plt.title('FFT-magnitude_FID', fontsize=15)
#plt.imshow(Dream_FID*mask_zero,vmin=0,vmax=0.083,origin='lower'); plt.axis('off'); plt.colorbar()


# %% S8: STEAM flip angle / B1 map
fig=plt.figure();
plt.subplot(); plt.title('B1 map', fontsize=15)
B1_angle = np.arctan(np.sqrt(2*(np.abs(img_STE))/(np.abs(img_FID))))*(180/np.pi)/55
Dream_b1=np.abs(B1_angle)*mask_zero       # created Zunayed
plt.imshow(np.abs(B1_angle)*mask_NaN, vmin=0, vmax=1.2, origin='lower'); plt.axis('off'); plt.colorbar() # 0 to 1.2 when comparing to WASABI  # 0 to 1.09 for simulation

#%%   marge all different type of B1 image   #################

# input data
#train18_B1=np.reshape(Dream_b1*mask_zero, (4096,1))               # chnage 18
#np.save('train_B1.npy', train_B1)

#marge_B1= np.load('marge_B1.npy')
#marge_B1=np.concatenate((marge_B1, train18_B1), axis=1)            # chnage 18
#np.save('marge_B1.npy', marge_B1)

# %% S9: B0 maps
fig=plt.figure();
plt.subplot(); plt.title('B0 phase map [rad]', fontsize=15)
B0_phase = np.angle(img_FID*np.conjugate(img_STE))
plt.imshow(B0_phase*mask_NaN,vmin=-np.pi,vmax=np.pi, origin='lower'); plt.colorbar()

# unwrap B0 map:
fig=plt.figure();
plt.subplot(); plt.title('B0 phase map unwrapped [rad]', fontsize=15)
B0_unwrap = unwrap_phase(B0_phase*mask_zero)
plt.imshow(B0_unwrap*mask_NaN,vmin=-np.pi,vmax=np.pi, origin='lower'); plt.colorbar()

# convert to Hz
fig=plt.figure();
plt.subplot(); plt.title('B0 map [Hz]', fontsize=15)
TE = 7*t0+gz3.fall_time
#B0_freq = B0_unwrap/(2*np.pi)*(1/(2*TE))
B0_freq = B0_phase/(2*np.pi)*(1/(2*TE))
Dream_b0=B0_freq*mask_zero                        # created zunayed###
plt.imshow(B0_freq*mask_NaN, vmin=-45, vmax=45, origin='lower'); plt.axis('off'); plt.colorbar()

#%%   marge all different type of B0 image   ################

# input data
#train18_B0=np.reshape(Dream_b0*mask_zero, (4096,1))           # chnage 18
#np.save('train_B0.npy', train_B0) 

#marge_B0= np.load('marge_B0.npy')
#marge_B0=np.concatenate((marge_B0, train18_B0), axis=1)            # chnage 18
#np.save('marge_B0.npy', marge_B0)

#%% craeate phantom 

# B0
#phan18_B0=np.reshape(B0*mask_zero, (4096,1))         # chnage 18
#np.save('phan_B0.npy', phan_B0)  

#marge_P_B0= np.load('marge_P_B0.npy')
#marge_P_B0=np.concatenate((marge_P_B0, phan18_B0), axis=1)   #  # chnage 18
#np.save('marge_P_B0.npy', marge_P_B0)


# B1
#phan18_B1=np.reshape(B1*mask_zero, (4096,1))         # chnage 18
#np.save('phan_B1.npy', phan_B1)
  
#marge_P_B1= np.load('marge_P_B1.npy')
#marge_P_B1=np.concatenate((marge_P_B1, phan18_B1), axis=1)   #  # chnage 18
#np.save('marge_P_B1.npy', marge_P_B1)


# %% S10: transceive chain maps
fig=plt.figure();
plt.subplot(); plt.title('transceive chain phase [rad]', fontsize=15)
txrx_phase = np.angle(img_FID*img_STE)
plt.imshow(txrx_phase*mask_NaN,vmin=-np.pi,vmax=np.pi, origin='lower'); plt.axis('off'); plt.colorbar()

fig=plt.figure();
plt.subplot(); plt.title('transceive chain phase unwrapped [rad]', fontsize=15)
txrx_unwrap = unwrap_phase(txrx_phase*mask_zero)
#plt.imshow(txrx_unwrap*mask_NaN,vmin=0,vmax=3*np.pi, origin='lower'); plt.colorbar()
#if unwrap starts in the wrong area and makes everything negative:
plt.imshow((txrx_unwrap-np.min(txrx_unwrap))*mask_NaN,vmin=0,vmax=3*np.pi, origin='lower'); plt.axis('off'); plt.colorbar()

# %% S11: compare DREAM simulation to simulation phantom

if use_simulation:
    fig=plt.figure();
    #B0 map sim phantom
    plt.subplot(231); plt.title('True/Phan B0 [Hz]', fontsize=15)
    plt.imshow(B0*mask_NaN, vmin=-41, vmax=37, origin="lower") 
    plt.axis('off')
    plt.colorbar()
        
    #B1 map sim phantom
    plt.subplot(234); plt.title('True/Phan B1 [a.u.]', fontsize=15)
    plt.imshow(B1*mask_NaN, vmin=0, vmax=1.09, origin="lower") 
    plt.axis('off')
    plt.colorbar()
    
    #B0 map DREAM sim
    plt.subplot(232); plt.title('STE DREAM B0 [Hz]', fontsize=15)
    plt.imshow(B0_freq*mask_NaN,vmin=-41,vmax=37,origin='lower')
    plt.axis('off')
    plt.colorbar()
    
    #B1 map DREAM sim
    plt.subplot(235); plt.title('STE DREAM B1 [a.u.]', fontsize=15)
    plt.imshow(np.abs(B1_angle)*mask_NaN, vmin=0 ,vmax=1.09, origin='lower')
    plt.axis('off')
    plt.colorbar()
    
    # B0 diff map
    plt.subplot(233); plt.title('STE B0 - True B0 [Hz]', fontsize=15)
    plt.imshow(-(B0-(B0_freq*mask_NaN)), vmin=-5, vmax=5, origin="lower") 
    plt.axis('off')
    plt.colorbar()
    '''
    #B1 diff map
    plt.subplot(247); plt.title('B1 diff map (DREAM-phantom)', fontsize=15)
    plt.imshow(-(B1-(np.abs(B1_angle)*mask_NaN)), vmin=-0.05, vmax=0.05, origin="lower")
    plt.colorbar()
    '''
    '''
    # B0 ratio map
    plt.subplot(244); plt.title('B0 ratio map (DREAM/phantom)', fontsize=15)
    plt.imshow((B0_freq*mask_NaN)/B0, vmin=-2, vmax=2, origin="lower") 
    plt.colorbar()
    '''
    #B1 ratio map
    plt.subplot(236); plt.title('STE B1 / True B1', fontsize=15)
    plt.imshow((np.abs(B1_angle)*mask_NaN)/B1, vmin=0.95, vmax=1.05, origin="lower") 
    plt.axis('off')
    plt.colorbar()
    
    
    #B0 scatter plot
    fig=plt.figure(figsize=(10,6));
    plt.subplot(111); plt.title('B0 comparison', fontsize=15)
    n = np.linspace(-55, 55, 111)
    slope1, intercept1, _, _, _ = linregress(np.reshape(B0*mask_zero,4096),np.reshape((B0_freq*mask_zero),4096))
    plt.xlabel('True/Phan B0 [Hz]', fontsize=15)
    plt.ylabel('STE DREAM B0 [Hz]', fontsize=15)
    plt.xlim(-55,55)
    plt.ylim(-55,55)
    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(np.reshape(B0,4096),np.reshape((B0_freq*mask_NaN),4096),'x', label='data')
    plt.plot(n, n, color='0.7', linestyle='--', label='optimal match')
    plt.plot(n, slope1*n+intercept1, 'r', label='linear fit')
    plt.legend(fontsize=15)
    
    #B1 scatter plot
    fig=plt.figure(figsize=(10,6));
    plt.subplot(111); plt.title('B1 comparison', fontsize=15)
    n = np.linspace(0, 1.2, 121)
    slope2, intercept2, _, _, _ = linregress(np.reshape(B1*mask_zero,4096),np.reshape((np.abs(B1_angle)*mask_zero),4096))
    plt.ylabel('STE DREAM B1 [a.u.]', fontsize=15)
    plt.xlabel('True/Phan B1 [a.u.]', fontsize=15)
    plt.xlim(0.6,1.2)
    plt.ylim(0.6,1.2)
    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(np.reshape(B1,4096),np.reshape((np.abs(B1_angle)*mask_NaN),4096),'x', label='data')
    plt.plot(n, n, color='0.7', linestyle='--', label='optimal match')
    plt.plot(n, slope2*n+intercept2, 'r', label='linear fit')
    plt.legend(fontsize=15)
    

# %% S12: WASABI
# essentially the same approach as for simulation comparison
if use_simulation==0: 
    
    # load .mat file
    mat = scipy.io.loadmat('E:/Dream_MR0_core/data/invivo_230620/WASABI.mat')

    # extract values from mat dictionary
    header = mat['__header__']
    version = mat['__version__']
    mat_globals = mat['__globals__']
    B1map = mat['B1map'] 
    dB0_stack_ext= mat['dB0_stack_ext'] # B0 map [ppm]
    M0_stack_wasabi= mat['M0_stack_wasabi']
    Mz_stack_wasabi= mat['Mz_stack_wasabi']
    P = mat['P']
    Z_wasabi = mat['Z_wasabi']
    popt = mat['popt']


    # resize function to get same resoultion as DREAM images
    def res(im, out_x, out_y):
        m = max(im.shape[0], im.shape[1])
        y = np.linspace(0, 1.0/m, im.shape[0])
        x = np.linspace(0, 1.0/m, im.shape[1])
        interpolating_function = RegularGridInterpolator((y, x), im)
        
        yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))
        
        return interpolating_function((xv, yv))


    # some constants and calculations
    B0_seq = 2.89362 # [T]  
    gamma_ = 42.5764 # [MHz/T]
    FREQ = B0_seq*gamma_ # scanner frequency [MHz]
    dB0_stack_ext = np.rot90(dB0_stack_ext,3)
    B0_Hz = dB0_stack_ext*FREQ # B0 map [Hz]
    B1map = np.rot90(B1map,3)



    # B1 and B0 maps from WASABI in original resolution
    fig = plt.figure() 
    
    plt.subplot(221); plt.title('B1 map WASABI', fontsize=15)
    plt.imshow(B1map, vmin=0, vmax=1.2, origin='lower'); plt.colorbar()
    
    plt.subplot(222); plt.title('B0 map WASABI [ppm]', fontsize=15)
    plt.imshow(dB0_stack_ext, vmin=-0.3, vmax=0.3, origin='lower'); plt.colorbar()

    plt.subplot(224); plt.title('B0 map WASABI [Hz]', fontsize=15)
    plt.imshow(B0_Hz, vmin=-45, vmax=45, origin='lower'); plt.colorbar()


    # B1 and B0 maps from WASABI in resized resolution matching DREAM
    B1map_res=res(B1map,64,64) #64,64
    dB0_stack_ext_res=res(dB0_stack_ext,64,64)
    B0_Hz_res=res(B0_Hz,64,64)
    
    # save wasabi image  created by zunayed
    wasabi_B0_img =B0_Hz_res     #created by zunayed
    wasabi_B1_img =B1map_res    #created by zunayed
   
    # plot image
    fig = plt.figure() 
    
    plt.subplot(221); plt.title('B1 map WASABI resized', fontsize=15)
    plt.imshow(B1map_res*mask_NaN, vmin=0, vmax=1.2, origin='lower'); plt.colorbar() #*mask_NaN
    
    plt.subplot(222); plt.title('B0 map WASABI [ppm] resized', fontsize=15)
    plt.imshow(dB0_stack_ext_res*mask_NaN, vmin=-0.3, vmax=0.3, origin='lower'); plt.colorbar()
    
    plt.subplot(224); plt.title('B0 map WASABI [Hz] resized', fontsize=15)
    plt.imshow(B0_Hz_res*mask_NaN, vmin=-45, vmax=45, origin='lower'); plt.colorbar()
    '''
    torch.save(B1map_res,'WASABI_B1')
    torch.save(dB0_stack_ext_res,'WASABI_B0_ppm')
    torch.save(B0_Hz_res,'WASABI_B0_Hz')
    '''
    # compare WASABI and DREAM
    fig = plt.figure() 
    
    plt.subplot(231); plt.title('WASABI B0 [Hz]', fontsize=15)
    plt.imshow(B0_Hz_res*mask_NaN, vmin=-45, vmax=45, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(234); plt.title('WASABI B1 [a.u.]', fontsize=15)
    plt.imshow(B1map_res*mask_NaN, vmin=0, vmax=1.2, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(232); plt.title('STE DREAM B0 [Hz]', fontsize=15)
    plt.imshow(B0_freq*mask_NaN, vmin=-45, vmax=45, origin='lower'); plt.axis('off'); plt.colorbar()

    plt.subplot(235); plt.title('STE DREAM B1 [a.u.]', fontsize=15)
    plt.imshow(np.abs(B1_angle)*mask_NaN, vmin=0, vmax=1.2, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(233); plt.title('STE B0 - WASABI B0 [Hz]', fontsize=15)
    plt.imshow((B0_freq-B0_Hz_res)*mask_NaN, vmin=-15, vmax=15, origin='lower'); plt.axis('off'); plt.colorbar()
    '''
    plt.subplot(247); plt.title('B1 diff DREAM-WASABI')
    plt.imshow((np.abs(B1_angle).numpy()-B1map_res)*mask_NaN, vmin=-0.2, vmax=0.1, origin='lower'); plt.colorbar()
    '''
    '''
    plt.subplot(244); plt.title('B0 ratio DREAM/WASABI')
    plt.imshow((B0_freq/B0_Hz_res)*mask_NaN, vmin=-8, vmax=8, origin='lower'); plt.colorbar()
    '''
    plt.subplot(236); plt.title('STE B1 / WASABI B1', fontsize=15)
    plt.imshow((np.abs(B1_angle)/B1map_res)*mask_NaN, vmin=0.8, vmax=1.1, origin='lower'); plt.axis('off'); plt.colorbar()
    
    
    #B0 scatter plot
    fig=plt.figure(figsize=(10,6));
    plt.subplot(111); plt.title('B0 comparison', fontsize=15)
    n = np.linspace(-100, 200, 301)
    slope3, intercept3, _, _, _ = linregress(np.reshape(B0_Hz_res*mask_zero,4096),np.reshape((B0_freq*mask_zero),4096))
    plt.ylabel('STE DREAM B0 [Hz]', fontsize=15)
    plt.xlabel('WASABI B0 [Hz]', fontsize=15)
    plt.xlim(-55,55) #-55,55
    plt.ylim(-55,55) #-55,55
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.plot(np.reshape(B0_Hz_res*mask_NaN,4096),np.reshape((B0_freq*mask_NaN),4096),'x',label='data')
    plt.plot(n, n, color='0.7', linestyle='--', label='optimal match')
    plt.plot(n, slope3*n+intercept3, 'r', label='linear fit')
    plt.legend(fontsize=15)
    
    
    #B1 scatter plot
    fig=plt.figure(figsize=(10,6));
    plt.subplot(111); plt.title('B1 comparison', fontsize=15)
    n = np.linspace(0.2, 1.6, 121)
    slope4, intercept4, _, _, _ = linregress(np.reshape(B1map_res*mask_zero,4096),np.reshape((np.abs(B1_angle)*mask_zero),4096))
    plt.ylabel('STE DREAM B1 [a.u.]', fontsize=15)
    plt.xlabel('WASABI B1 [a.u.]', fontsize=15)
    plt.xlim(0.6,1.2) #0,1.2
    plt.ylim(0.6,1.2) #0,1.2
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.plot(np.reshape(B1map_res*mask_zero,4096),np.reshape((np.abs(B1_angle)*mask_zero),4096),'x',label='data')
    plt.plot(n, n, color='0.7', linestyle='--', label='optimal match')
    plt.plot(n, slope4*n+intercept4, 'r', label='linear fit')
    plt.legend(fontsize=15)
    
    

# %% S13: export / import 

#exporting or importing maps as arrays (used when you want to compare STE and STID maps)

exp = 0
imp = 0

if exp:
    if use_simulation:
        torch.save(B1_angle,'STE_B1_sim')
    else:
        torch.save(img_STE,'invivo_STE_imgSTE_p13_230615')
        torch.save(img_FID,'invivo_STE_imgFID_p13_230615')
        torch.save(B1_angle,'invivo_STE_B1_p13_230615')
        torch.save(B0_phase,'invivo_STE_B0phase_p13_230615')
        torch.save(B0_freq,'invivo_STE_B0freq_p13_230615')
        torch.save(txrx_phase,'invivo_STE_txrxphase_p13_230615')
    
if imp:
    if use_simulation:
        B1_angle_STID=torch.load('STID_B1_sim')
    else:
        #B1_angle_STID=torch.load('STID_B1_inv')
        img_STE_13=torch.load('phantom_STE_imgSTE_p13_230615')
        img_FID_13=torch.load('phantom_STE_imgFID_p13_230615')
        B1_angle_13=torch.load('phantom_STE_B1_p13_230615')
        B0_phase_13=torch.load('phantom_STE_B0phase_p13_230615')
        B0_freq_13=torch.load('phantom_STE_B0freq_p13_230615')
        txrx_phase_13=torch.load('phantom_STE_txrxphase_p13_230615')
        

# %% S14: compare STE and STID
# load B1 map from STID timing and compare to STE B1 map
if 0: #imp
    fig = plt.figure() 
    
    plt.subplot(131); plt.title('STID DREAM B1 [a.u.]', fontsize=15)
    plt.imshow(np.abs(B1_angle_STID)*mask_NaN, vmin=0, vmax=1.2, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(132); plt.title('STE DREAM B1 [a.u.]', fontsize=15)
    plt.imshow(np.abs(B1_angle)*mask_NaN, vmin=0, vmax=1.2, origin='lower'); plt.axis('off'); plt.colorbar()
    '''
    plt.subplot(143); plt.title('B1 diff map STID-STE', fontsize=15)
    plt.imshow((np.abs(B1_angle_STID)-np.abs(B1_angle))*mask_NaN, vmin=-0.05, vmax=0.05, origin='lower'); plt.axis('off'); plt.colorbar()
    '''
    plt.subplot(133); plt.title('STID B1 / STE B1', fontsize=15)
    plt.imshow((np.abs(B1_angle_STID)/np.abs(B1_angle))*mask_NaN, vmin=0.9, vmax=1.1, origin='lower'); plt.axis('off'); plt.colorbar()
    
    
    fig=plt.figure(figsize=(10,6));
    plt.subplot(111);# plt.title('B1 comparison', fontsize=15)
    n = np.linspace(0.2, 1.6, 121)
    slope5, intercept5, _, _, _ = linregress(np.reshape((np.abs(B1_angle_STID)*mask_zero),4096),np.reshape(np.abs(B1_angle)*mask_zero,4096))
    plt.xlabel('STID B1 [a.u.]', fontsize=15)
    plt.ylabel('STE B1 [a.u.]', fontsize=15)
    plt.xlim(0.6,1.2) #0,1.2
    plt.ylim(0.6,1.2) #0,1.2
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.plot(np.reshape((np.abs(B1_angle_STID)*mask_NaN),4096),np.reshape((np.abs(B1_angle)*mask_NaN),4096),'x', label='data')
    plt.plot(n, n, color='0.7', linestyle='--', label='optimal match')
    plt.plot(n, slope5*n+intercept5, 'r', label='linear fit')
    plt.legend(fontsize=15)
    

# %% S15: compare pulseq v1.3 and v1.2
if imp:
    fig = plt.figure()

    plt.subplot(221); plt.title('FFT-magnitude_STE_p1.3', fontsize=15) # all channels
    plt.imshow(np.abs(img_STE_13), vmin=0, vmax=8.6e-6, origin='lower'); plt.axis('off'); plt.colorbar() # , origin='lower' for reversed y axis
    plt.subplot(223); plt.title('FFT-magnitude_FID_p1.3', fontsize=15) # all channels
    plt.imshow(np.abs(img_FID_13), vmin=0, vmax=8.6e-6, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(222); plt.title('FFT-phase_STE_p1.3 [rad]', fontsize=15) # all channels 
    plt.imshow(np.angle(img_STE_13), vmin=-np.pi, vmax=np.pi, origin='lower'); plt.axis('off'); plt.colorbar()
    plt.subplot(224); plt.title('FFT-phase_FID_p1.3 [rad]', fontsize=15) # all channels
    plt.imshow(np.angle(img_FID_13), vmin=-np.pi, vmax=np.pi, origin='lower'); plt.axis('off'); plt.colorbar()
    
    
    fig = plt.figure()

    plt.subplot(231); plt.title('FFT-magnitude_STE_p1.3', fontsize=15) # all channels
    plt.imshow(np.abs(img_STE_13), vmin=0, vmax=8.6e-6, origin='lower'); plt.axis('off'); plt.colorbar() # , origin='lower' for reversed y axis
    plt.subplot(234); plt.title('FFT-magnitude_FID_p1.3', fontsize=15) # all channels
    plt.imshow(np.abs(img_FID_13), vmin=0, vmax=8.6e-6, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(232); plt.title('FFT-magnitude_STE_p1.2', fontsize=15) # all channels
    plt.imshow(np.abs(img_STE), vmin=0, vmax=8.6e-6, origin='lower'); plt.axis('off'); plt.colorbar() # , origin='lower' for reversed y axis
    plt.subplot(235); plt.title('FFT-magnitude_FID_p1.2', fontsize=15) # all channels
    plt.imshow(np.abs(img_FID), vmin=0, vmax=8.6e-6, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(233); plt.title('mag_STE p1.3 - p1.2', fontsize=15) # all channels
    plt.imshow(np.abs(img_STE_13)-np.abs(img_STE), vmin=-1.5e-6, vmax=1.5e-6, origin='lower'); plt.axis('off'); plt.colorbar() # , origin='lower' for reversed y axis
    plt.subplot(236); plt.title('mag_FID p1.3 - p1.2', fontsize=15) # all channels
    plt.imshow(np.abs(img_FID_13)-np.abs(img_FID), vmin=-1.5e-6, vmax=1.5e-6, origin='lower'); plt.axis('off'); plt.colorbar()
    
    
    fig = plt.figure()
    
    plt.subplot(131); plt.title('B1 pulseq1.3 [a.u.]', fontsize=15) 
    plt.imshow(np.abs(B1_angle_13)*mask_NaN, vmin=0, vmax=1.2, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(132); plt.title('B1 pulseq1.2 [a.u.]', fontsize=15) 
    plt.imshow(np.abs(B1_angle)*mask_NaN, vmin=0, vmax=1.2, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(133); plt.title('B1 p1.3 / B1 p1.2', fontsize=15) 
    plt.imshow((np.abs(B1_angle_13)/np.abs(B1_angle))*mask_NaN, vmin=0.9, vmax=1.1, origin='lower'); plt.axis('off'); plt.colorbar()
    
    
    fig = plt.figure()
    
    plt.subplot(131); plt.title('B0 pulseq1.3 [rad]', fontsize=15)
    plt.imshow(B0_phase_13*mask_NaN,vmin=-np.pi,vmax=np.pi, origin='lower'); plt.colorbar()
    
    plt.subplot(132); plt.title('B0 pulseq1.2 [rad]', fontsize=15)
    plt.imshow(B0_phase*mask_NaN,vmin=-np.pi,vmax=np.pi, origin='lower'); plt.colorbar()
    
    plt.subplot(133); plt.title('B0 p1.3 - B0 p1.2 [rad]', fontsize=15)
    plt.imshow((B0_phase_13-B0_phase)*mask_NaN,vmin=-0.2,vmax=0.2, origin='lower'); plt.colorbar()
    
    
    fig = plt.figure()
    
    plt.subplot(131); plt.title('txrx pulseq1.3 [rad]', fontsize=15)
    plt.imshow(txrx_phase_13*mask_NaN,vmin=-np.pi,vmax=np.pi, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(132); plt.title('txrx pulseq1.2 [rad]', fontsize=15)
    plt.imshow(txrx_phase*mask_NaN,vmin=-np.pi,vmax=np.pi, origin='lower'); plt.axis('off'); plt.colorbar()
    
    plt.subplot(133); plt.title('txrx p1.3 - txrx p1.2 [rad]', fontsize=15)
    plt.imshow((txrx_phase_13-txrx_phase)*mask_NaN,vmin=-0.2,vmax=0.2, origin='lower'); plt.axis('off'); plt.colorbar()

#%%

               ####    END Dream sequence Code   #####
    ####################################################################################################     
               

#%%

             ####  Save data for   Dream Sequence Linear Regession and Neural Network Model   ####
    ####################################################################################################
             
    
#%% Create train Large dataset:  generate 20 images from 20 deferent Dream sequences 

# =============================================================================
# ### save 20 FID image  data
# 
# train18_FID=np.reshape(Dream_FID*mask_zero, (4096,1))               
# np.save('train_FID.npy', train_FID)
# marge_FID= np.load('marge_FID.npy')
# marge_FID=np.concatenate((marge_FID, train18_FID), axis=1)    # concateante all image     
# np.save('marge_FID.npy', marge_FID)
# 
# ###  save 20 STE image data
# 
# train18_STE=np.reshape(Dream_STE*mask_zero, (4096,1))              
# np.save('train_STE.npy', train_STE)
# marge_STE= np.load('marge_STE.npy')
# marge_STE=np.concatenate((marge_STE, train18_STE), axis=1)    # concateante all image        
# np.save('marge_STE.npy', marge_STE)
# 
# ###  Plot FID and STE image   
#      
# fig=plt.figure()
# plt.subplot(121); plt.title('FFT-magnitude_STE', fontsize=15)
# plt.imshow(Dream_STE*mask_zero,vmin=0,vmax=0.083,origin='lower'); plt.axis('off'); plt.colorbar()
# plt.subplot(122); plt.title('FFT-magnitude_FID', fontsize=15)
# plt.imshow(Dream_FID*mask_zero,vmin=0,vmax=0.083,origin='lower'); plt.axis('off'); plt.colorbar()
# 
# ###  save 20 BO image data
# 
# train18_B0=np.reshape(Dream_b0*mask_zero, (4096,1))           
# np.save('train_B0.npy', train_B0) 
# marge_B0= np.load('marge_B0.npy')
# marge_B0=np.concatenate((marge_B0, train18_B0), axis=1)      # concateante all image      
# np.save('marge_B0.npy', marge_B0)
# 
# ###  save 20 B1 image data
# 
# train18_B1=np.reshape(Dream_b1*mask_zero, (4096,1))              
# np.save('train_B1.npy', train_B1)
# marge_B1= np.load('marge_B1.npy')
# marge_B1=np.concatenate((marge_B1, train18_B1), axis=1)     # concateante all image       
# np.save('marge_B1.npy', marge_B1)
# =============================================================================


#%% Create Target/ Phantom image for respective each of train data set:  generate target 20 images 

# =============================================================================
# ### save 20 target BO image data
# phan18_B0=np.reshape(B0*mask_zero, (4096,1))        
# np.save('phan_B0.npy', phan_B0)  
# 
# marge_P_B0= np.load('marge_P_B0.npy')
# marge_P_B0=np.concatenate((marge_P_B0, phan18_B0), axis=1)   # concateante all image
# np.save('marge_P_B0.npy', marge_P_B0)
# 
# 
# ### save 20 target B1 image data
# phan18_B1=np.reshape(B1*mask_zero, (4096,1))         
# np.save('phan_B1.npy', phan_B1)
#   
# marge_P_B1= np.load('marge_P_B1.npy')
# marge_P_B1=np.concatenate((marge_P_B1, phan18_B1), axis=1)   # concateante all image
# np.save('marge_P_B1.npy', marge_P_B1)
# 
# =============================================================================


#%%   Save and Load image mask

####   Load mask_zero and 
#np.save('mask_zero.npy', mask_zero) 
#mask_zero= np.load('mask_zero.npy')

#np.save('mask_NaN.npy', mask_NaN) 
#mask_NaN= np.load('mask_NaN.npy')

#%% Defind WASABI image for future Neural network

#wasabi_B0_img =B0_Hz_res     #created by zunayed
#wasabi_B1_img =B1map_res    #created by zunayed

#%% save vivo image and wasabi image

# =============================================================================
# # # test  vivo data
# vivo_B0 = np.reshape(B0_freq , (4096,1))                                   
# vivo_B1 = np.reshape(np.abs(B1_angle).numpy(), (4096,1))                      
# vivo_FID= np.reshape(Dream_vivo_FID.numpy(), (4096,1))
# vivo_STE=np.reshape( Dream_vivo_STE.numpy(), (4096,1))
# vivo_all= np. concatenate((vivo_B0, vivo_B1, vivo_FID, vivo_STE), axis=1)
# np.save('vivo_all.npy', vivo_all)
# 
# wasabi_B0 = np.reshape( wasabi_B0_img, (4096,1))                                
# wasabi_B1 = np.reshape(  wasabi_B1_img, (4096,1))
# wasabi_all= np. concatenate((wasabi_B0, wasabi_B1), axis=1)
# np.save('wasabi_all.npy', wasabi_all)
# =============================================================================

# =============================================================================
# # save vivo mask
# mask_NaN_vivo=mask_NaN
# np.save('mask_NaN_vivo.npy', mask_NaN_vivo)
# mask_zero_vivo=mask_zero
# np.save('mask_zero_vivo.npy', mask_zero_vivo)
# =============================================================================
