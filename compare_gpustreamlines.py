import argparse
import random
import time

import numpy as np

import dipy.reconst.dti as dti
from dipy.io import read_bvals_bvecs
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking import utils
from dipy.core.gradients import gradient_table
from dipy.data import small_sphere
from dipy.reconst.shm import OpdtModel, CsaOdfModel
from dipy.reconst import shm
from dipy.data import get_fnames
from dipy.data import read_stanford_pve_maps
from dipy.direction import BootDirectionGetter
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

import nibabel as nib
from nibabel.orientations import aff2axcodes

from dipy.tracking.utils import density_map
from dipy.io.utils import get_reference_info
from dipy.segment.bundles import bundle_adjacency
from dipy.io.streamline import load_tractogram
from dipy.tracking import Streamlines
from dipy.tracking.streamline import set_number_of_points

# Import custom module
import cuslines.cuslines as cuslines

#Get Gradient values
def get_gtab(fbval, fbvec):
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    return gtab

def get_img(ep2_seq):
    img = nib.load(ep2_seq)
    return img

def dice_score(A,B):
    
    img_sum = np.add(A,B)
    denom = np.sum(img_sum)
    
    img_interx = np.copy(img_sum)
    img_interx[img_interx!=2] = 0
    
    numer = np.sum(img_interx)
    
    return np.float(numer)/np.float(denom)

def recall(GT,PREDICTED):
    
    img_sum = np.add(GT,PREDICTED)
    denom = np.sum(GT)
    
    TP = np.copy(img_sum)
    TP[TP!=2] = 0
    TP[TP>0] = 1
    
    numer = np.sum(TP)
    
    return np.float(numer)/np.float(denom)

def precision(GT,PREDICTED):
    
    img_sum = np.add(GT,PREDICTED)
    denom = np.sum(PREDICTED)
    
    TP = np.copy(img_sum)
    TP[TP!=2] = 0
    TP[TP>0] = 1
    
    numer = np.sum(TP)
    
    return np.float(numer)/np.float(denom)

def compare_voxelwise(trk1, trk2):
    
    # read trk files
    sft1 = load_tractogram(trk1, 'same', bbox_valid_check=False)
    sft1.remove_invalid_streamlines()
    s1 = Streamlines(sft1.streamlines)

    sft2 = load_tractogram(trk2, 'same', bbox_valid_check=False)
    sft2.remove_invalid_streamlines()
    s2 = Streamlines(sft2.streamlines)

    # get voxel-wise density maps
    affine1, dimensions1, voxel_sizes1, voxel_order1 = get_reference_info(trk1)
    dm_1 = density_map(s1, affine1, dimensions1)

    affine2, dimensions2, voxel_sizes2, voxel_order2 = get_reference_info(trk2)
    dm_2 = density_map(s2, affine2, dimensions2)

    # binarize
    dm_1[dm_1>0]=1
    dm_1[dm_1<1]=0
    
    dm_2[dm_2>0]=1
    dm_2[dm_2<1]=0

    return dm_1, dm_2

def compare_buan(trk1, trk2, threshold=5):
    
    # read trk files
    sft1 = load_tractogram(trk1, 'same', bbox_valid_check=False)
    sft1.remove_invalid_streamlines()
    s1 = Streamlines(sft1.streamlines)

    sft2 = load_tractogram(trk2, 'same', bbox_valid_check=False)
    sft2.remove_invalid_streamlines()
    s2 = Streamlines(sft2.streamlines)

    # set points
    n_pts = 20

    bundle1 = set_number_of_points(s1,n_pts)
    bundle2 = set_number_of_points(s2,n_pts)

    ba_score = bundle_adjacency(bundle1,bundle2,threshold=threshold)

    return ba_score

# input parameters
fa_threshold = 0.1
nseeds = 20000
sampling_density = 3
sh_order = 4
sm_lambda = 0.006
min_signal = 0.1 
max_angle = 60
sh_order = 4
relative_peak_threshold = 0.25
min_separation_angle = 45
step_size = 0.5

# Get Stanford HARDI data
hardi_nifti_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
csf, gm, wm = read_stanford_pve_maps()
wm_data = wm.get_fdata()
img = get_img(hardi_nifti_fname)
voxel_order = "".join(aff2axcodes(img.affine))
gtab = get_gtab(hardi_bval_fname, hardi_bvec_fname)

data = img.get_fdata()
roi_data = (wm_data > 0.5)
mask = roi_data

tenmodel = dti.TensorModel(gtab, fit_method='WLS')
print('Fitting Tensor')
tenfit = tenmodel.fit(data, mask)
print('Computing anisotropy measures (FA,MD,RGB)')
FA = tenfit.fa
FA[np.isnan(FA)] = 0

# Setup tissue_classifier args
tissue_classifier = ThresholdStoppingCriterion(FA, fa_threshold)
metric_map = np.asarray(FA, 'float64')

# Create seeds for ROI
seed_mask = utils.seeds_from_mask(roi_data, density=sampling_density, affine=np.eye(4))
seed_mask = seed_mask[0:nseeds]

# model
model = CsaOdfModel(gtab, sh_order=sh_order, smooth=sm_lambda, min_signal=min_signal)

# Run CPU tracking 

# set seed to get deterministic streamlines
np.random.seed(0)
random.seed(0)

# Setup direction getter
boot_dg = BootDirectionGetter.from_data(data, model, max_angle=max_angle, sphere=small_sphere, sh_order=sh_order, relative_peak_threshold=relative_peak_threshold, min_separation_angle=min_separation_angle)

streamline_generator = LocalTracking(boot_dg, tissue_classifier, seed_mask, affine=np.eye(4), step_size=step_size)
streamlines = [s for s in streamline_generator]

fname_cpu = 'compare_CPU.trk'

sft1 = StatefulTractogram(streamlines, hardi_nifti_fname, Space.VOX)
save_tractogram(sft1, fname_cpu)

# Run CPU tracking again

# do not reset seed
#np.random.seed(0)
#random.seed(0)

boot_dg = BootDirectionGetter.from_data(data, model, max_angle=max_angle, sphere=small_sphere, sh_order=sh_order, relative_peak_threshold=relative_peak_threshold, min_separation_angle=min_separation_angle)

streamline_generator = LocalTracking(boot_dg, tissue_classifier, seed_mask, affine=np.eye(4), step_size=step_size)
streamlines2 = [s for s in streamline_generator]

fname_cpu2 = 'compare_CPU2.trk'

sft2 = StatefulTractogram(streamlines2, hardi_nifti_fname, Space.VOX)
save_tractogram(sft2, fname_cpu2)

# GPUStreamlines

# set seed to get deterministic streamlines
np.random.seed(0)
random.seed(0)

# model
#model = CsaOdfModel(gtab, sh_order=args.sh_order, smooth=args.sm_lambda, min_signal=args.min_signal)
fit_matrix = model._fit_matrix
# Unlike OPDT, CSA has a single matrix used for fit_matrix. Populating delta_b and delta_q with necessary values for
# now.
delta_b = fit_matrix
delta_q = fit_matrix

b0s_mask = gtab.b0s_mask
dwi_mask = ~b0s_mask

sphere = small_sphere
theta = sphere.theta
phi = sphere.phi
sampling_matrix, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)

x, y, z = model.gtab.gradients[dwi_mask].T
r, theta, phi = shm.cart2sphere(x, y, z)
B, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)
H = shm.hat(B)
R = shm.lcr_matrix(H)

# create floating point copy of data
dataf = np.asarray(data, dtype=float)

gpu_tracker = cuslines.GPUTracker(cuslines.ModelType.CSAODF,
                                  max_angle * np.pi/180,
                                  min_signal,
                                  fa_threshold,
                                  step_size,
                                  relative_peak_threshold,
                                  min_separation_angle * np.pi/180,
                                  dataf, H, R, delta_b, delta_q,
                                  b0s_mask.astype(np.int32), metric_map, sampling_matrix,
                                  sphere.vertices, sphere.edges.astype(np.int32),
                                  ngpus=1, rng_seed=0)

streamlines_gpu = gpu_tracker.generate_streamlines(seed_mask)

sft_gpu = StatefulTractogram(streamlines_gpu, hardi_nifti_fname, Space.VOX)
fname_gpu = 'compare_GPU.trk'
save_tractogram(sft_gpu, fname_gpu)

# Compare streamline results

print('Number of streamlines CPU-1: ', len(streamlines))
print('Number of streamlines CPU-2: ', len(streamlines2))
print('Number of streamlines GPU-1: ', len(streamlines_gpu))

# Compare CPU vs CPU-2
dm_cpu_1, dm_cpu_2 = compare_voxelwise(fname_cpu, fname_cpu2)

print('Dice score CPU-1 vs CPU-2: ',dice_score(dm_cpu_1,dm_cpu_2))
print('Precision score CPU-1 vs CPU-2: ',precision(dm_cpu_1,dm_cpu_2))
print('Recall score CPU-1 vs CPU-2: ',recall(dm_cpu_1,dm_cpu_2))

ba_score_cpu = compare_buan(fname_cpu, fname_cpu2)

print('BUAN score CPU-1 vs CPU-2: ', ba_score_cpu)

# Compare CPU vs GPU
dm_cpu_1, dm_gpu = compare_voxelwise(fname_cpu, fname_gpu)

print('Dice score CPU-1 vs GPU: ',dice_score(dm_cpu_1,dm_gpu))
print('Precision score CPU-1 vs GPU: ',precision(dm_cpu_1,dm_gpu))
print('Recall score CPU-1 vs GPU: ',recall(dm_cpu_1,dm_gpu))

ba_score_gpu = compare_buan(fname_cpu, fname_gpu)

print('BUAN score CPU-1 vs GPU: ', ba_score_gpu)