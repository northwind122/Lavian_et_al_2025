import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates
from matplotlib import colors


def to_255_range(a, quantile_min=0.001, quantile_max=0.999):
    vmin, vmax = np.quantile(a, [quantile_min, quantile_max])
    return ((np.clip(a, vmin, vmax)-vmin)*255/(vmax-vmin)).astype(np.uint8)


def show_slices(img, i=None, j=None, k=None):
    i, j, k = [d if d is not None else s//2 for s, d in zip(img.shape, [i, j ,k])]
    full_sliceview = np.zeros((img.shape[0] + img.shape[2], img.shape[1] + img.shape[2]), img.dtype)
    full_sliceview[img.shape[2]:, :img.shape[1]] = img[:, :, k]
    full_sliceview[:img.shape[2], :img.shape[1]] = img[i, :, :].T
    full_sliceview[img.shape[2]:, img.shape[1]:] = img[:, j, :]
    return Image.fromarray(full_sliceview)
   
    
def show_mip(img):
    full_sliceview = np.zeros((img.shape[0] + img.shape[2], img.shape[1] + img.shape[2]), img.dtype)
    full_sliceview[img.shape[2]:, :img.shape[1]] = np.max(img, axis=2)
    full_sliceview[:img.shape[2], :img.shape[1]] = np.max(img, axis=0).T
    full_sliceview[img.shape[2]:, img.shape[1]:] = np.max(img, axis=1)
    return Image.fromarray(full_sliceview)


def map_affine(stack,  transform_mat, dest_shape, order=3):
    target_coords = np.reshape(transform_mat @ np.reshape(
        np.pad(np.indices(dest_shape), ((0,1), (0,0), (0,0), (0,0)),
        mode="constant", constant_values=1), (4, -1)), 
                                (3, *dest_shape))

    return map_coordinates(stack, target_coords, order=order)


def get_slice(stack, index, dimension):
    if dimension == 0:
        return(stack[index, :, :])
    elif dimension == 1:
        return(stack[:, index, :])
    elif dimension == 2:
        return(stack[:, :, index])
    

def plot_overlay(ref, mov, plane, dim, ref_title='Reference', mov_title='Warped movout'):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    
    ref_slice = get_slice(ref, plane, dim)
    mov_slice = get_slice(mov, plane, dim)
    
    axes[0].imshow(ref_slice, cmap="gray", origin='lower')
    axes[0].set_title(ref_title, c='green')
    
    composite = np.stack([(mov_slice-np.min(mov_slice))/(np.max(mov_slice)-np.min(mov_slice)), 
                          (ref_slice-np.min(ref_slice))/(np.max(ref_slice)-np.min(ref_slice)), 
                          (mov_slice-np.min(mov_slice))/(np.max(mov_slice)-np.min(mov_slice))], 2)
    axes[1].imshow(composite, origin='lower')
    axes[1].set_title('Overlay')
    
    axes[2].imshow(mov_slice, cmap="gray", origin='lower')
    axes[2].set_title(mov_title, c='magenta')

    
def plot_side_to_side(stack1, stack2, depth, dim, stack1_title='Stack 1', stack2_title='Stack 2'):
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    stack_1_slice = get_slice(stack1, int(np.round((stack1.shape[dim]-1)*depth/100)), dim)
    stack_2_slice = get_slice(stack2, int(np.round((stack2.shape[dim]-1)*depth/100)), dim)

    axes[0].imshow(stack_1_slice, origin='lower', cmap='gray_r')
    axes[0].set_title('{} ({}%)'.format(stack1_title, depth))

    axes[1].imshow(stack_2_slice, origin='lower', cmap='gray_r')
    axes[1].set_title('{} ({}%)'.format(stack2_title, depth));
    
    
def view_jacobian(ref, mov_affine, mov_warped, jacobian_mat, plane, dim):
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    
    ref_slice = get_slice(ref, plane, dim)
    mov_aff_slice = get_slice(mov_affine, plane, dim)
    mov_warp_slice = get_slice(mov_warped, plane, dim)
    jacobian_slice = get_slice(jacobian_mat, plane, dim)

    composite1 = np.stack([(mov_aff_slice-np.min(mov_aff_slice))/(np.max(mov_aff_slice)-np.min(mov_aff_slice)), 
                           (ref_slice-np.min(ref_slice))/(np.max(ref_slice)-np.min(ref_slice)), 
                           (mov_aff_slice-np.min(mov_aff_slice))/(np.max(mov_aff_slice)-np.min(mov_aff_slice))], 2)
    
    composite2 = np.stack([(mov_warp_slice-np.min(mov_warp_slice))/(np.max(mov_warp_slice)-np.min(mov_warp_slice)), 
                           (ref_slice-np.min(ref_slice))/(np.max(ref_slice)-np.min(ref_slice)), 
                           (mov_warp_slice-np.min(mov_warp_slice))/(np.max(mov_warp_slice)-np.min(mov_warp_slice))], 2)

    axes[0, 0].imshow(mov_aff_slice, cmap='gray_r')
    axes[0, 0].set_title('Pre warping')
    axes[0, 0].set_ylabel('Mov stack')
    axes[1, 0].imshow(composite1)
    axes[1, 0].set_ylabel('Overlay')
    
    axes[0, 1].imshow(jacobian_slice, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('Jacobian det.')
    axes[1, 1].imshow(jacobian_slice, cmap='RdBu_r', vmin=-1, vmax=1)

    axes[0, 2].imshow(mov_warp_slice, cmap='gray_r')
    axes[0, 2].set_title('Post warping')
    axes[1, 2].imshow(composite2)

    plt.tight_layout()
