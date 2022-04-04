#################
#### IMPORTS ####
#################

# Arrays
import numpy as np
import cytoolz
import dask as ds
from dask.array import Array as daskarr
import dask.array as da

# Deep Learning stuff
import torch
import torchvision
import torchvision.transforms as transforms

# Images display and plots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl

# Fancy progress bars
import tqdm.notebook as tq

# Tensor Network Stuff
import quimb.tensor as qtn # Tensor Network library
import quimb

from collections import deque

# Profiling
import cProfile, pstats, io
from pstats import SortKey

import functools
import collections
import opt_einsum as oe
import itertools
import pydash as _pd
import copy
import os
#######################################################
'''
Wrapper for type checks.
While defining a function, you can add the wrapper
stating the expected types:
> @arg_val(class_1, class_2, ...)
> def function(a, b, ...): 
'''
def arg_val(*args):
    def wrapper(func):
        def validating(*_args):
            if any(type(_arg)!=arg for _arg, arg in zip(_args,args)):
                raise TypeError('wrong type!')
            return func(*_args)
        return validating
    return wrapper

def pro_profiler(func):
    '''Generic profiler. Expects an argument-free function.
    e. g. func = lambda: learning_epoch_SGD(mps, imgs, 3, 0.1).
    Prints and returns the profiling report trace.'''
    # TODO: adapt to write trace to file
    pr = cProfile.Profile()
    pr.enable()
    func()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    return s

class mps_lr:
    def __init__(self, mps, lr0, momentum, s_factor):
        self.sites = len(mps.tensors)
        self.lr0 = lr0
        self.curr_lr = self.lr0
        self.momentum = momentum
        self.t = 0
        self.s_factor = s_factor
        self.past_grad = (self.sites-1)*[0]
        
    
    def clear(self):
        self.t = 0
    
    def compute_lr(self, t):
        '''
        Lr will be annealed each epoch through an exponential function that
        takes into account the baseline value
        '''
        return ((self.lr0)*np.exp(-self.s_factor*self.t  ) )
    
    def new_epoch(self):
        '''
        Update curr lr to the next timestep
        '''
        self.t = self.t + 1
        self.curr_lr = self.compute_lr(self.t)
    
    def J(self, left_index, dNLL):
        '''
        J^{k+1} = \beta J^{k} + dNLL^{k}
        A^{k+1} = A^{k} - lr* J^{k+1}
        '''
        if self.t > 0:
            J = self.momentum * self.past_grad[left_index] + dNLL
        else:
            J = dNLL
        
        self.past_grad[left_index] = np.mean(dNLL.data)/dNLL.data.max()
        
        return J
            
        
#   ___     
#  |_  |    
#   _| |_ _ 
#  |_____|_| MNIST FUNCTIONS
#######################################################        

def get_data(train_size = 1000, test_size = 100, grayscale_threshold = .5):
    '''
    Prepare the MNIST dataset for the training algorithm:
     * Choose randomly a subset from the whole dataset
     * Flatten each image to mirror the mps structure
     * Normalize images from [0,255] to [0,1]
     * Apply a threshold for each pixels so that each value 
       below that threshold are set to 0, the others get set to 1.
       For this algorithm we will only deal to binary states {0,1}
       instead of a range from 0 to 1    
    '''
    # Download all data
    mnist = torchvision.datasets.MNIST('classifier_data', train=True, download=True,
                                                  transform = transforms.Compose([transforms.ToTensor()]) )
    
    # Convert torch.tenor to numpy
    npmnist = mnist.data.numpy()
    
    # Check of the type of the sizes
    #if ((type(train_size) != int) or (type(test_size) != int)):
    #    raise TypeError('train_size and test_size must be INT')
    
    # Check if the training_size and test_size requested are bigger than
    # the MNIST whole size
    if ( (train_size + test_size) > npmnist.shape[0] ):
        raise ValueError('Subset too big') 
    
    # Check of the positivity of sizes
    if ( (train_size <= 0) or (test_size <= 0) ):
        raise ValueError('Size of training set and test set cannot be negative')
    
    # Choose just a subset of the data
    # Creating a mask by randomly sampling the indexes of the full dataset
    subset_indexes = np.random.choice(np.arange(npmnist.shape[0]), size=(train_size + test_size), 
                                      replace=False, p=None)
    
    # Apply the mask
    npmnist = npmnist[subset_indexes]

    # Flatten every image
    npmnist = np.reshape(npmnist, (npmnist.shape[0], npmnist.shape[1]*npmnist.shape[2]))
    
    # Normalize the data from 0 - 255 to 0 - 1
    npmnist = npmnist/npmnist.max()
    
    # As in the paper, we will only deal with {0,1} values, not a range
    
    if ((grayscale_threshold <= 0) or (grayscale_threshold >= 1)):
        raise ValueError('grayscale_threshold must be in range ]0,1[')
    
    npmnist[npmnist > grayscale_threshold] = 1
    npmnist[npmnist <= grayscale_threshold] = 0
    
    # Return training set and test set
    return npmnist[:train_size], npmnist[train_size:]

def plot_img(img_flat, shape, flip_color = True, border = False, savefig = '', title=''):
    '''
    Display the image from the flattened form
    '''
    # If the image is corrupted for partial reconstruction (pixels are set to -1)
    if -1 in img_flat:
        img_flat = np.copy(img_flat)
        img_flat[img_flat == -1] = 0
    plt.figure(figsize = (2,2))
    
    if title != '':
        plt.title(title)
    # Background white, strokes black
    if flip_color:
        plt.imshow(1-np.reshape(img_flat,shape), cmap='gray')
    # Background black, strokes white
    else:
        plt.imshow(np.reshape(img_flat,shape), cmap='gray')
        
    if border:
        plt.xticks([])
        plt.yticks([])
    else:
        plt.axis('off')
    
    if savefig != '':
        # save the picture as svg in the location determined by savefig
        plt.savefig(savefig, format='svg')
        plt.show()
        
def partial_removal_img(mnistimg, shape, fraction = .5, axis = 0, half = None):
    '''
    Corrupt (with -1 values) a portion of an input image (from the test set)
    to test if the algorithm can reconstruct it
    '''
    # Check shape:
    if len(shape) != 2 or (shape[0]<1 or shape[1]<1):
        raise ValueError('The shape of an image needs two positive integer components')
    # Check type:
    if [type(mnistimg), type(fraction), type(axis)] != [np.ndarray, float, int]:
        raise TypeError('Input types not valid')
    
    # Check the shape of input image
    if (mnistimg.shape[0] != shape[0]*shape[1]):
        raise TypeError(f'Input image shape does not match, need (f{shape[0]*shape[1]},)')
    
    # Axis can be either 0 (rowise deletion) or 1 (columnwise deletion)
    if not(axis in [0,1]):
        raise ValueError('Invalid axis [0,1]')
    
    # Fraction must be from 0 to 1
    if (fraction < 0 or fraction > 1):
        raise ValueError('Invalid value for fraction variable (in interval [0,1])')
        
    mnistimg_corr = np.copy(mnistimg)
    mnistimg_corr = np.reshape(mnistimg_corr, shape)
    
    if half == None:
        half = np.random.randint(2)
    
    if axis == 0:
        if half == 0:
            mnistimg_corr[int(shape[0]*(1-fraction)):,:] = -1
        else:
            mnistimg_corr[:int(shape[0]*(1-fraction)),:] = -1
    else:
        if half == 0:
            mnistimg_corr[:,int(shape[1]*(1-fraction)):] = -1
        else:
            mnistimg_corr[:,:int(shape[1]*(1-fraction))] = -1
        
    mnistimg_corr = np.reshape(mnistimg_corr, (shape[0]*shape[1],))
    
    return mnistimg_corr

def plot_rec(cor_flat, rec_flat, shape, savefig = '', N = 2):
    '''
    Display the RECONSTRUCTION
    '''
    
    # PREPARING CMAPS
    greycmap = pl.cm.Greys

    # Get the colormap colors
    corrupted_cmap = greycmap(np.arange(greycmap.N))

    # Set alpha
    corrupted_cmap[:,-1] = np.linspace(0, 1, greycmap.N)

    # Create new colormap
    corrupted_cmap = ListedColormap(corrupted_cmap)
    
    reccolors = [(1, 0, 0), (1, 1, 1)]
    reccmap = LinearSegmentedColormap.from_list('rec', reccolors, N=N)
    
    # If the image is corrupted for partial reconstruction (pixels are set to -1)
    cor_flat = np.copy(cor_flat)
    cor_flat[cor_flat == -1] = 0
    plt.figure(figsize = (2,2))
    
    plt.imshow(1-np.reshape(rec_flat, shape), cmap=reccmap)
    plt.imshow(np.reshape(cor_flat, shape), cmap=corrupted_cmap)
        
    plt.axis('off')
    
    if savefig != '':
        # save the picture as svg in the location determined by savefig
        plt.savefig(savefig, format='svg')
    
    plt.show()
        

#   ____    
#  |___ \   
#    __) |  
#   / __/ _ 
#  |_____(_) MPS GENERAL
#######################################################

@functools.lru_cache(2**12)
def _inds_to_eq(inputs, output):
    """
    Conert indexes to the equation of contractions for np.einsum function
    """
    symbol_get = collections.defaultdict(map(oe.get_symbol, itertools.count()).__next__).__getitem__
    in_str = ("".join(map(symbol_get, inds)) for inds in inputs)
    out_str = "".join(map(symbol_get, output))
    return ",".join(in_str) + f"->{out_str}"

def tneinsum(tn1,tn2):
    '''
    Contract tn1 with tn2
    It is an automated function that automatically contracts the bonds with
    the same indexes.
    For simple contractions this function is faster than tensor_contract
    or @
    '''
    inds_i = tuple([tn1.inds, tn2.inds])
    inds_out = tuple(qtn.tensor_core._gen_output_inds(cytoolz.concat(inds_i)))
    eq = qtn.tensor_core._inds_to_eq(inds_i, inds_out)
    
    data = np.einsum(eq,tn1.data,tn2.data)
    
    return qtn.Tensor(data=data, inds=inds_out)

def tneinsum2(tn1,tn2):
    '''
    Contract tn1 with tn2
    It is an automated function that automatically contracts the bonds with
    the same indexes.
    For simple contractions this function is faster than tensor_contract
    or @
    '''
    inds_i = tuple([tn1.inds, tn2.inds])
    inds_out = tuple(qtn.tensor_core._gen_output_inds(cytoolz.concat(inds_i)))
    eq = _inds_to_eq(inds_i, inds_out)
    
    data = np.einsum(eq,tn1.data,tn2.data)
    
    return qtn.Tensor(data=data, inds=inds_out)

@functools.lru_cache(2**12)
def arr_inds_to_eq(inputs, output):
    """
    Conert indexes to the equation of contractions for np.einsum function
    """
    symbol_get = collections.defaultdict(map(oe.get_symbol, itertools.count()).__next__).__getitem__
    in_str = ("".join(map(symbol_get, inds)) for inds in inputs)
    out_str = "".join(map(symbol_get, output))
    return "i"+",i".join(in_str) + f"->i{out_str}"

def into_data(tensor_array):
    return np.array([ten.data.astype(np.float32) for ten in tensor_array])

def _into_data(tensor_array):
    op_arr = []
    for ten in tensor_array:
        op_arr.append(ds.delayed(lambda x: x.data.astype(np.float32))(ten))
    data_arr = ds.delayed(lambda x: x)(op_arr).compute()
    return data_arr

def into_tensarr(data_arr,inds):
    return np.array([qtn.Tensor(data=data,inds=inds) for data in data_arr])

def tneinsum3(*tensor_lists,backend = 'torch'):
    '''
    Takes arrays of tensors and contracts them element by element.
    '''
    # Retrieve indeces from the first elements
    inds_in = tuple([arr[0].inds for arr in tensor_lists])
    # Output indeces
    inds_out = tuple(qtn.tensor_core._gen_output_inds(cytoolz.concat(inds_in)))
    # Convert into einsum expression with extra index for entries
    eq = arr_inds_to_eq(inds_in, inds_out)
    # Generate a list of arrays of numpy tensors
    tens_data = [into_data(ten) for ten in tensor_lists]
    # Extract the shapes
    shapes = [tens.shape for tens in tens_data]
    # prepare opteinsum reduction expression
    expr = oe.contract_expression(eq,*shapes)
    # execute and extract
    data_arr = expr(*tens_data,backend = backend)

    return into_tensarr(data_arr,inds_out)

def initialize_mps(Ldim, bdim = 30, canonicalize = 0):
    '''
    Initialize the MPS tensor network
    1. Create the MPS TN
    2. Canonicalization
    3. Renaming indexes
    '''
    # Create a simple MPS network randomly initialized
    mps = qtn.MPS_rand_state(Ldim, bond_dim=bdim)
    
    # Canonicalize: use a canonicalize value out of range to skip it (such as -1)
    if canonicalize in range(Ldim):
        mps.canonize(canonicalize)
        
    # REINDEXING TENSORS FOR A EASIER DEVELOPING
    # during initializations, the index will be named using the same notation of the 
    # Pan Zhang et al paper:
    #  ___       ___                      ___
    # |I0|--i0--|I1|--i1-... ...-i(N-1)--|IN|
    #  |         |                        |
    #  | v0      | v1                     | vN
    #  V         V                        V
    
    # Reindexing the leftmost tensor
    mps = mps.reindex({mps.tensors[0].inds[0]: 'i0', 
                       mps.tensors[0].inds[1]: 'v0'})
    
    # Reindexing the inner tensors through a cycle
    for tensor in range(1,len(mps.tensors)-1):
        mps = mps.reindex({mps.tensors[tensor].inds[0]: 'i'+str(tensor-1),
                           mps.tensors[tensor].inds[1]: 'i'+str(tensor),
                           mps.tensors[tensor].inds[2]: 'v'+str(tensor)})
    
    # Reindexing the last tensor
    tensor = tensor + 1
    mps = mps.reindex({mps.tensors[tensor].inds[0]: 'i'+str(tensor-1),
                       mps.tensors[tensor].inds[1]: 'v'+str(tensor)})  
    
    return mps

def quimb_transform_img2state(img):
    '''
    Trasform an image to a tensor network to fully manipulate
    it using quimb, may be very slow, use it for checks
    '''
    
    # Initialize empty tensor
    img_TN = qtn.Tensor()
    for k, pixel in enumerate(img):
        if pixel == 0: # if pixel is 0, we want to have a tensor with data [0,1]
            img_TN = img_TN &  qtn.Tensor(data=[0,1], inds=['v'+str(k)], )
            
        else: # if pixel is 1, we want to have a tensor with data [1,0]
            img_TN = img_TN &  qtn.Tensor(data=[1,0], inds=['v'+str(k)], )
     
    # |  | 781 |
    # O  O ... O
    return img_TN


def stater(x,i):
    if x in [0,1]:
        vec = [int(x), int(not x)]
        return qtn.Tensor(vec,inds=(f'v{i}',))
    return None

def tens_picture(picture):
    '''Converts an array of bits into a list of tensors compatible with a tensor network.'''
    tens = [stater(n,i) for i, n in enumerate(picture)]
    return np.array(tens)

def left_right_cache(mps,_imgs):
    curr_l = np.array(len(_imgs)*[qtn.Tensor()])
    curr_l = curr_l.reshape((len(_imgs),1))
    for site in range(len(mps.tensors)-1):
        machines = np.array(len(_imgs)*[mps[site]])
        contr_l = tneinsum3(curr_l[:,-1],machines,_imgs[:,site])
        data = into_data(contr_l)
        maxa = np.abs(data).max(axis = tuple(range(1,len(data.shape))))
        data = data/maxa.reshape((len(_imgs),1))
        contr_l = into_tensarr(data,contr_l[0].inds)
        contr_l = contr_l.reshape((len(_imgs),1))
        curr_l = np.hstack([curr_l,contr_l])
    curr_r = np.array(len(_imgs)*[qtn.Tensor()])
    curr_r = curr_r.reshape((len(_imgs),1))
    for site in range(len(mps.tensors)-1,0,-1):
        machines = np.array(len(_imgs)*[mps[site]])
        contr_r = tneinsum3(curr_r[:,0],machines,_imgs[:,site])
        data = into_data(contr_r)
        maxa = np.abs(data).max(axis = tuple(range(1,len(data.shape))))
        data = data/maxa.reshape((len(_imgs),1))
        contr_r = into_tensarr(data,contr_r[0].inds)
        contr_r = contr_r.reshape((len(_imgs),1))
        curr_r = np.hstack([contr_r,curr_r])
    img_cache = np.array([curr_l,curr_r]).transpose((1,0,2))
    return img_cache

def ext_left_right_cache(mps,_imgs):
    # WARNING: THIS IS EXTREMELY SLOW.
    # It's more convenient to initialize on RAM and convert to dask array
    curr_l = da.from_array(len(_imgs)*[qtn.Tensor()], chunks = (len(_imgs)))
    curr_l = curr_l.reshape((len(_imgs),1))
    for site in range(len(mps.tensors)-1):
        machines = np.array(len(_imgs)*[mps[site]])
        contr_l = tneinsum3(curr_l[:,-1].compute(),machines,_imgs[:,site])
        contr_l = da.from_array(contr_l.reshape((len(_imgs),1)),chunks = (len(_imgs)))
        curr_l = da.hstack((curr_l,contr_l))
    curr_r = da.from_array(len(_imgs)*[qtn.Tensor()], chunks = (len(_imgs)))
    curr_r = curr_r.reshape((len(_imgs),1))
    for site in range(len(mps.tensors)-1,0,-1):
        machines = np.array(len(_imgs)*[mps[site]])
        contr_r = tneinsum3(curr_r[:,0].compute(),machines,_imgs[:,site])
        contr_r = da.from_array(contr_r.reshape((len(_imgs),1)),chunks = (len(_imgs)))
        curr_r = da.hstack((contr_r,curr_r))
    img_cache = da.from_array([curr_l,curr_r],chunks = (1,len(_imgs),1)).transpose((1,0,2))
    return img_cache


def sequential_update(mps,_imgs,img_cache,site,going_right):
    if type(img_cache) == daskarr:
        return ext_sequential_update(mps,_imgs,img_cache,site,going_right)
    if going_right:
        left_cache = img_cache[:,0,site]
        left_imgs = _imgs[:,site]
        new_cache = tneinsum3(left_cache,np.array(len(_imgs)*[mps[site]]),left_imgs)
        data = into_data(new_cache)
        axes = tuple(range(1,len(data.shape)))
        maxa = np.abs(data).max(axis = axes)
        data = data/maxa.reshape((len(_imgs),1))
        new_cache = into_tensarr(data,new_cache[0].inds)
        img_cache[:,0,site+1] = new_cache
    else:
        right_cache = img_cache[:,1,site+1]
        right_imgs = _imgs[:,site+1]
        new_cache = tneinsum3(right_cache,np.array(len(_imgs)*[mps[site+1]]),right_imgs)
        data = into_data(new_cache)
        axes = tuple(range(1,len(data.shape)))
        maxa = np.abs(data).max(axis = axes)
        data = data/maxa.reshape((len(_imgs),1))
        new_cache = into_tensarr(data,new_cache[0].inds)
        img_cache[:,1,site] = new_cache

def ext_sequential_update(mps,_imgs,img_cache,site,going_right):
    if going_right:
        left_cache = img_cache[:,0,site].compute()
        left_imgs = _imgs[:,site]
        new_cache = tneinsum3(left_cache,np.array(len(_imgs)*[mps[site]]),left_imgs)
        img_cache[:,0,site+1] = new_cache
    else:
        right_cache = img_cache[:,1,site+1].compute()
        right_imgs = _imgs[:,site+1]
        new_cache = tneinsum3(right_cache,np.array(len(_imgs)*[mps[site+1]]),right_imgs)
        img_cache[:,1,site] = new_cache

def restart_cache(mps,site,left_cache,right_cache,_img):
    left_site = site
    right_site = site + 1
    left_cache[:site+1] = extend_cache(mps,qtn.Tensor(),_img,0,left_site)
    right_cache[site+1:] = np.flip(extend_cache(mps,qtn.Tensor(),_img,len(_img)-1,right_site))
    return left_cache,right_cache

def advance_cache(mps,left_cache,right_cache,going_right,curr_site,last_site,_img):
    if going_right:
        state = left_cache[last_site]
        left_cache[last_site:curr_site+1] = extend_cache(mps,state,_img,last_site,curr_site)
    else:
        state = right_cache[last_site+1]
        right_cache[curr_site+1:last_site+2] = np.flip(extend_cache(mps,state,_img,last_site+1,curr_site+1))
    return left_cache, right_cache

def extend_cache(mps,base,_img,start,end):
    out = [base]
    step = int(end>=start)-int(end<start)
    guide = np.arange(start,end,step)
    for index in guide:
        A, qubit = mps[index],_img[index]
        # First, contract the state, then the qubit
        # TODO: is qtn.tensor_contract(out[-1],A,qubit) faster?
        out.append(tneinsum2(tneinsum2(out[-1],A),qubit))
    return out # Verify sorting

def half_cache(mps,right_cache,left_cache,last_site,curr_site,going_right,_img):
    '''
    Applied when we were going in the opposite direction half an epoch ago.
    Restarts one cache, and extend the other if required
    '''
    if going_right:
        # This means we were going left last time, therefore we have to recreate the whole left cache
        left_cache[:curr_site+1] = extend_cache(mps,qtn.Tensor(),_img,0,curr_site)
        if curr_site < last_site:
            state = right_cache[last_site+1]
            right_cache[curr_site+1:last_site+2] = np.flip(extend_cache(mps,state,_img,last_site+1,curr_site+1))
    else:
        # We were going right and are now going left. Restart right cache
        right_cache[curr_site+1:] = np.flip(extend_cache(mps,qtn.Tensor(),_img,len(_img)-1,curr_site+1))
        if curr_site > last_site:
            state = left_cache[last_site]
            left_cache[last_site:curr_site+1] = extend_cache(mps,state,_img,last_site,curr_site)
    return left_cache, right_cache

def stochastic_cache_update(mps,_imgs,img_cache,last_dirs,last_sites,last_epochs,mask,going_right,curr_epoch,curr_site):
    '''
    each last_x array is a size len(img_cache) array which specifies x for the last update of the image at the given position
    last_dir: specifices the direction we were heading when we last updated each image cache
    last_site: specifies the last index site for which we updated the image cache
    last_epoch: specifies the epoch during which we last updated the image cache
    mask: mask array of images whose cache we wish to use.
    '''
    for index in mask:
        _img = _imgs[index]
        left_cache, right_cache = img_cache[index]
        last_epoch = last_epochs[index]
        went_right = last_dirs[index]
        last_site = last_sites[index]
        if curr_epoch > last_epoch:#(curr_epoch assumed to be >= last_epoch)
            if last_epoch == -1:
                left_cache, right_cache = restart_cache(mps, curr_site, left_cache, right_cache, _img)
            elif (curr_epoch == last_epoch + 1) and (going_right>went_right):
                # (It is assumed that the first stage is going right, and the second, going left)
                # If we are in the first stage of the current epoch,
                # and we were at the second stage of the last one,
                # we build the left cache from the initial site,
                # and correspondingly rescale the right cache, which is still useful.
                left_cache, right_cache = half_cache(mps,right_cache,left_cache,last_site,curr_site,going_right,_img)
            else:
                # In this scenario, we must recreate everything
                left_cache, right_cache = restart_cache(mps, curr_site, left_cache, right_cache, _img)
        elif going_right<went_right:
            # (we cannot be in the same epoch and going right if last time we were going left)
            # We're then in the same epoch
            # if we're now going left and were going right, we need to create the right cache from the last site
            # and correspondingly rescale the left cache, which is still useful
            left_cache, right_cache = half_cache(mps,right_cache,left_cache,last_site,curr_site,going_right,_img)
        else: # We're then going in the same direction as last time
            # So we must grow the corresponding site and shorten the other
            left_cache, right_cache = advance_cache(mps,left_cache, right_cache, going_right,curr_site,last_site,_img)
        last_sites[index] = curr_site
        last_epochs[index] = curr_epoch
        last_dirs[index] = going_right
        img_cache[index] = left_cache, right_cache

def computepsi(mps, img):
    '''
    Contract the MPS with the states (pixels) of a binary{0,1} image
    
    PSI:    O-...-O-O-O-...-O
            |     | | |     |
            |     | | |     |
    IMAGE:  O     O O O     O
    
    Images state are created the following way:
    if pixel is 0 -> state = [0,1]
    if pixel is 1 -> state = [1,0]
    '''
    
    # Left most tensor
    #          O--
    # Compute  |  => O--
    #          O
    if img[0] == 0:
        contraction = np.einsum('a,ba',[0,1], mps.tensors[0].data)
    else:
        contraction = np.einsum('a,ba',[1,0], mps.tensors[0].data)
        
    # Remove the first and last pixels because in the MPS
    # They need to be treated differently
    for k, pixel in enumerate(img[1:-1]):
        #  
        # Compute  O--O--  => O--
        #             |       |
        contraction = np.einsum('a,abc',contraction, mps.tensors[k+1].data)
        
        #          O--
        # Compute  |  => O--
        #          O        
        if pixel == 0:
            contraction = np.einsum('a,ba', [0,1], contraction)
        else:
            contraction = np.einsum('a,ba', [1,0], contraction)
    
    #          
    # Compute  O--O  => O
    #             |     |
    contraction = np.einsum('a,ab',contraction, mps.tensors[-1].data)
    
    #          O
    # Compute  |  => O (SCALAR)
    #          O     
    if img[-1] == 0:
        contraction = np.einsum('a,a', [0,1], contraction)
    else:
        contraction = np.einsum('a,a', [1,0], contraction)
    
    return contraction

def computepsiprime(mps, img, contracted_left_index):
    '''
    Contract the MPS with the states (pixels) of a binary{0,1} image
    
    PSI':    O-...-O-      -O-...-O
             |     |        |     |
             |     |  |  |  |     |
    IMAGE:   O     O  O  O  O     O
    
    Images state are created the following way:
    if pixel is 0 -> state = [0,1]
    if pixel is 1 -> state = [1,0]
    '''
    
    if contracted_left_index == 0:
        ##############
        # RIGHT PART #
        ##############

        # Right most tensor
        #          ---O
        # Compute     |  => --O
        #             O
        if img[-1] == 0:
            contraction_dx = np.einsum('a,ba',[0,1], mps.tensors[-1].data)
        else:
            contraction_dx = np.einsum('a,ba',[1,0], mps.tensors[-1].data)

        for k in range(len(mps.tensors)-2, contracted_left_index+1, -1):
            #  
            # Compute  --O--O  => --O
            #               |       |

            contraction_dx = np.einsum('a,bac->bc',contraction_dx, mps.tensors[k].data)

            #          --O
            # Compute    |  => --O
            #            O        
            if img[k] == 0:
                contraction_dx = np.einsum('a,ba', [0,1], contraction_dx)
            else:
                contraction_dx = np.einsum('a,ba', [1,0], contraction_dx)
                
        # From here on it is just speculation

        if img[contracted_left_index] == 0:
            v0 = [0,1]
        else:
            v0 = [1,0]

        if img[contracted_left_index+1] == 0:
            contraction_dx = np.einsum('a,k->ak', contraction_dx, [0,1])
        else:
            contraction_dx = np.einsum('a,k->ak', contraction_dx, [1,0])

        contraction = np.einsum('a,cd->acd', v0, contraction_dx)

        return contraction
    
    elif contracted_left_index == len(mps.tensors) - 2:
        #############
        # LEFT PART #
        #############

        # Left most tensor
        #          O--
        # Compute  |  => O--
        #          O
        if img[0] == 0:
            contraction_sx = np.einsum('a,ba',[0,1], mps.tensors[0].data)
        else:
            contraction_sx = np.einsum('a,ba',[1,0], mps.tensors[0].data)

        for k in range(1, contracted_left_index):
            #  
            # Compute  O--O--  => O--
            #             |       |
            contraction_sx = np.einsum('a,abc->bc',contraction_sx, mps.tensors[k].data)

            #          O--
            # Compute  |  => O--
            #          O        
            if img[k] == 0:
                contraction_sx = np.einsum('a,ba', [0,1], contraction_sx)
            else:
                contraction_sx = np.einsum('a,ba', [1,0], contraction_sx)
        
        # From here on it is just speculation

        if img[contracted_left_index] == 0:
            contraction_sx = np.einsum('a,k->ak', contraction_sx, [0,1])
        else:
            contraction_sx = np.einsum('a,k->ak', contraction_sx, [1,0])

        if img[contracted_left_index+1] == 0:
            vm1 = [0,1]
        else:
            vm1 = [1,0]

        contraction = np.einsum('ab,c->abc', contraction_sx, vm1)

        return contraction
    else:
        #############
        # LEFT PART #
        #############

        # Left most tensor
        #          O--
        # Compute  |  => O--
        #          O
        if img[0] == 0:
            contraction_sx = np.einsum('a,ba',[0,1], mps.tensors[0].data)
        else:
            contraction_sx = np.einsum('a,ba',[1,0], mps.tensors[0].data)

        for k in range(1, contracted_left_index):
            #  
            # Compute  O--O--  => O--
            #             |       |
            contraction_sx = np.einsum('a,abc->bc',contraction_sx, mps.tensors[k].data)

            #          O--
            # Compute  |  => O--
            #          O        
            if img[k] == 0:
                contraction_sx = np.einsum('a,ba', [0,1], contraction_sx)
            else:
                contraction_sx = np.einsum('a,ba', [1,0], contraction_sx)

        ##############
        # RIGHT PART #
        ##############

        # Right most tensor
        #          ---O
        # Compute     |  => --O
        #             O
        if img[-1] == 0:
            contraction_dx = np.einsum('a,ba',[0,1], mps.tensors[-1].data)
        else:
            contraction_dx = np.einsum('a,ba',[1,0], mps.tensors[-1].data)

        for k in range(len(mps.tensors)-2, contracted_left_index+1, -1):
            #  
            # Compute  --O--O  => --O
            #               |       |

            contraction_dx = np.einsum('a,bac->bc',contraction_dx, mps.tensors[k].data)

            #          --O
            # Compute    |  => --O
            #            O        
            if img[k] == 0:
                contraction_dx = np.einsum('a,ba', [0,1], contraction_dx)
            else:
                contraction_dx = np.einsum('a,ba', [1,0], contraction_dx)

        # From here on it is just speculation

        if img[contracted_left_index] == 0:
            contraction_sx = np.einsum('a,k->ak', contraction_sx, [0,1])
        else:
            contraction_sx = np.einsum('a,k->ak', contraction_sx, [1,0])

        if img[contracted_left_index+1] == 0:
            contraction_dx = np.einsum('a,k->ak', contraction_dx, [0,1])
        else:
            contraction_dx = np.einsum('a,k->ak', contraction_dx, [1,0])

        contraction = np.einsum('ab,cd->abcd', contraction_sx, contraction_dx)

        return contraction

def psi_primed(mps,_img,index):
    # quimby contraction. Currently not faster than einsum implementation
    # Requires _img to be provided in a quimby friendly format
    # Achievable with tens_picture
    contr_L = qtn.Tensor() if index == 0 else mps[0]@_img[0]
    for i in range(1,index,1):
        # first, contr_Lact with the other matrix
        contr_L = contr_L@mps[i]
        # then, with the qubit
        contr_L = contr_L@_img[i]
    contr_R = qtn.Tensor() if index == len(_img)-1 else mps[-1]@_img[-1]
    for i in range(len(_img)-2,index+1,-1):
        contr_R = mps[i]@contr_R
        contr_R = _img[i]@contr_R
    psi_p = contr_L@_img[index]@_img[index+1]@contr_R
    return psi_p

def computeNLL(mps, imgs, canonicalized_index = False):
    '''
    Computes the Negative Log Likelihood of a Tensor Network (mps)
    over a set of images (imgs)
    
     > NLL = -(1/|T|) * SUM_{v\in T} ( ln P(v) ) = -(1/|T|) * SUM_{v\in T} ( ln psi(v)**2 )
           = -(2/|T|) * SUM_{v\in T} ( ln |psi(v)| )
    '''
     
    if type(canonicalized_index) == int and 0<= canonicalized_index and canonicalized_index <= len(mps.tensors):
        Z = tneinsum2(mps.tensors[canonicalized_index], mps.tensors[canonicalized_index]).data
    else:
        Z = mps @ mps
        
    lnsum = 0   
    for img in imgs:
        lnsum = lnsum + np.log( abs(computepsi(mps,img)) )
        
    return - 2 * (lnsum / imgs.shape[0]) + np.log(Z)

def computeNLL_cached(mps, _imgs, img_cache, index):

    A = qtn.tensor_contract(mps[index],mps[index+1])
    Z = qtn.tensor_contract(A,A)

    psi_primed_arr =arr_psi_primed_cache(_imgs,img_cache,index)
    psi = tneinsum3(np.array(len(_imgs)*[A]),psi_primed_arr)
    logpsi = np.log(np.abs(into_data(psi)))
    sum_log = logpsi.sum()
    #             __    _        _         __  _    _           _        _
    # NLL = - _1_ \  ln|  _P(V)_  | = -_1_ \  |  ln| |Psi(v)|^2  | - lnZ  |
    #         |T| /_   |_   Z    _|    |T| /_ |_   |_           _|       _|
    #             _  __                      _         __
    #     = -_1_ | 2 \  ln|Psi(v)| - |T|lnZ   | = -_2_ \  ln|Psi(v)|  - lnZ
    #        |T| |_  /_                      _|    |T| /_
    return -(2/len(_imgs))*sum_log + np.log(Z)

def compress(mps, max_bond):
    
    for index in range(len(mps.tensors)-2,-1,-1):
        A = qtn.tensor_contract(mps[index],mps[index+1])

        if index == 0:
            SD = A.split(['v'+str(index)], absorb='left', max_bond = max_bond)
        else:
            SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='left',max_bond = max_bond)

        if index == 0:
            mps.tensors[index].modify(data=np.transpose(SD.tensors[0].data,(1,0)))
            mps.tensors[index+1].modify(data=SD.tensors[1].data)
        else:
            mps.tensors[index].modify(data=np.transpose(SD.tensors[0].data,(0,2,1)))
            mps.tensors[index+1].modify(data=SD.tensors[1].data)
    
    return mps

def compress_copy(mps, max_bond):
    comp_mps = copy.copy(mps)
    
    for index in range(len(comp_mps.tensors)-2,-1,-1):
        A = qtn.tensor_contract(comp_mps[index],comp_mps[index+1])

        if index == 0:
            SD = A.split(['v'+str(index)], absorb='left', max_bond = max_bond)
        else:
            SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='left',max_bond = max_bond)

        if index == 0:
            comp_mps.tensors[index].modify(data=np.transpose(SD.tensors[0].data,(1,0)))
            comp_mps.tensors[index+1].modify(data=SD.tensors[1].data)
        else:
            comp_mps.tensors[index].modify(data=np.transpose(SD.tensors[0].data,(0,2,1)))
            comp_mps.tensors[index+1].modify(data=SD.tensors[1].data)
    
    return comp_mps

def compress2(mps, max_bond):
    '''
    unlike copress function, this first checks if the bond between
    two tensors is higher than maxbond. If not it skips the pair.
    
    THIS FUNCTION BREAKS CANONIZATION but it is faster
    '''
    for index in range(len(mps.tensors)-2,-1,-1):
        if mps.bond_sizes()[index] > max_bond:
            A = qtn.tensor_contract(mps[index],mps[index+1])

            if index == 0:
                SD = A.split(['v'+str(index)], absorb='left', max_bond = max_bond)

                mps.tensors[index].modify(data=np.transpose(SD.tensors[0].data,(1,0)))
                mps.tensors[index+1].modify(data=SD.tensors[1].data)
            else:
                SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='left',max_bond = max_bond)

                mps.tensors[index].modify(data=np.transpose(SD.tensors[0].data,(0,2,1)))
                mps.tensors[index+1].modify(data=SD.tensors[1].data)
        else:
            pass
            
    return mps

def torchized_mps(mps,inds_dict):
    '''
    Expects a quimb tensor network and a dictionary to place the indeces.
    Returns an array with torch tensors for each of the mps sites.
    '''
    torch_mps = np.empty(len(mps.tensors),dtype = torch.Tensor)
    for site, tens in enumerate(mps.tensors):
        inds = tens.inds
        _tens = torch.from_numpy(np.array(tens.data,dtype = np.float32))
        inds_dict['mps'][site] = inds
        if torch.cuda.is_available():
            _tens = _tens.to('cuda')
        torch_mps[site] = _tens
        del _tens
    return torch_mps

def torchized_imgs(_imgs,inds_dict):
    '''
    Expects an array of quimb tensorized qubits.
    Turns it into an array of torch tensors.
    Theres one torch tensor per site,
    of length equal to the number of images.
    '''
    torch_imgs = np.empty(_imgs.shape[1],dtype = torch.Tensor)
    for site in range(_imgs.shape[1]):
        inds = _imgs[:,site][0].inds
        tens = torch.from_numpy(into_data(_imgs[:,site]))
        inds_dict['imgs'][site] = (inds)
        if torch.cuda.is_available():
            tens = tens.to('cuda')
        torch_imgs[site] = tens
        del tens
    return torch_imgs

def torch_contract(inds_in,*tensors):
    '''
    Takes arrays of tensors and contracts them given the indeces.
    Indeces are expected as a tuple of tuples.
    Each of the inner tuples describes the indeces that correspond to the tensor.
    ((i0,v1,i1),(i1,v2,i2))->(i0,v1,v2,i2)
    [A[:, :, :],B[:, :, :]]->C[:, :, :, :]
    '''
    # Output indeces
    inds_out = tuple(qtn.tensor_core._gen_output_inds(cytoolz.concat(inds_in)))
    # Convert into einsum expression
    eq = _inds_to_eq(inds_in, inds_out)
    # Extract the shapes
    shapes = [tens.shape for tens in tensors]
    # prepare opteinsum reduction expression
    expr = oe.contract_expression(eq,*shapes)
    # execute and extract
    data_arr = expr(*tensors,backend = 'torch')
    return data_arr,inds_out

def torch_multicontract(inds_in,*tensor_lists):
    '''
    Takes arrays of tensors and contracts them element by element.
    '''
    # Output indeces
    inds_out = tuple(qtn.tensor_core._gen_output_inds(cytoolz.concat(inds_in)))
    # Convert into einsum expression with extra index for entries
    eq = arr_inds_to_eq(inds_in, inds_out)
    # Extract the shapes
    shapes = [tens.shape for tens in tensor_lists]
    # prepare opteinsum reduction expression
    expr = oe.contract_expression(eq,*shapes)
    # execute and extract
    data_arr = expr(*tensor_lists,backend = 'torch')
    return data_arr,inds_out

def sequential_update_torched(torch_mps,torch_imgs,torch_cache,site,going_right,inds_dict):
    '''
    Updates the cache for the given site and direction.
    Can also be used to initialize cache.
    '''
    # TODO: exploit binary states
    # TODO: adapt to masked version for lighter usage

    # Direction-informed update arguments
    current = site + 1*(not going_right)
    target = site + 1*going_right
    side = 'left'*going_right+'right'*(not going_right)
    l = int(not going_right)

    size = torch_imgs[current].shape[0]

    tens_cache = torch_cache[0,l,current]
    tens_imgs = torch_imgs[current]

    # Broadcast mps for multicontraction. Usually way more efficient,
    # though memory heavy.
    tens_mps = torch.unsqueeze(torch_mps[current],0)[size*[0]]

    # Extract info on the current position to update the next.
    imgs_l_inds = inds_dict['imgs'][current]
    mps_l_inds = inds_dict['mps'][current]
    cache_inds = inds_dict[side][current]
    inds_in = [mps_l_inds,imgs_l_inds]
    tensors = [tens_mps,tens_imgs]
    tensors = [tens_cache] + tensors
    inds_in = [cache_inds] + inds_in

    # Perform multicontraction
    data, inds_out = torch_multicontract(tuple(inds_in),*tensors)

    # Rescaling because of overflows
    new_cache = data/torch.max(data,dim=1,keepdim = True)[0]

    # Try to place in the GPU
    if torch.cuda.is_available():
      new_cache = new_cache.to('cuda')
    inds_dict[side][target] = inds_out
    torch_cache[0,l,target] = new_cache
    del tens_cache,tens_imgs,tens_mps,tensors,new_cache
    torch.cuda.empty_cache()

def torchized_cache(torch_mps,torch_imgs,inds_dict):
    '''
    Initializes the torchized cache. Updates the indeces dictionary.
    '''
    size = torch_imgs[0].shape[0]
    pixels = len(torch_mps)
    torch_cache = np.empty(shape = (1,2,pixels),dtype = torch.Tensor)
    nully = qtn.Tensor()
    inds = nully.inds
    tons = np.array(size*[nully])
    tans = torch.from_numpy(into_data(tons))
    if torch.cuda.is_available():
        tans = tans.to('cuda')
    torch_cache[0,0,0] = tans
    torch_cache[0,1,-1] = tans
    inds_dict['left'][0] = inds
    inds_dict['right'][-1] = inds
    for site in range(pixels-2,-1,-1):
        sequential_update_torched(torch_mps,torch_imgs,torch_cache,site,False,inds_dict)
    del tans
    return torch_cache

def arr_psi_primed_torched(torch_imgs,torch_cache,index,mask,inds_dict):
    '''
    Computes the derivative of psi up to a constant for a mask-sliced collection of images.
    '''
    # Extract the cache and free legs
    left_cache = torch_cache[0,0,index][mask]
    right_cache = torch_cache[0,1,index+1][mask]
    left_imgs = torch_imgs[index][mask]
    right_imgs = torch_imgs[index+1][mask]

    # Extract the corresponding indeces
    left_inds = inds_dict['left'][index]
    right_inds = inds_dict['right'][index+1]
    img_l_inds = inds_dict['imgs'][index]
    img_r_inds = inds_dict['imgs'][index+1]

    # Place together, they're friends
    inds_in = [img_l_inds,img_r_inds]
    tensors = [left_imgs,right_imgs]

    # These don't always get along with the others...
    if index != 0:
        inds_in = [left_inds] + inds_in
        tensors = [left_cache] + tensors
    if index != len(torch_imgs)- 2:
        inds_in = inds_in + [right_inds]
        tensors = tensors + [right_cache]

    # Contract in parallel
    psi_primed_arr, inds_out = torch_multicontract(tuple(inds_in),*tensors)
    del left_cache,right_cache,left_imgs,right_imgs,tensors
    torch.cuda.empty_cache()
    return psi_primed_arr, inds_out

#   _____
#  |___ /
#    |_ \
#   ___) |
#  |____(_) LEARNING FUNCTIONS
#######################################################

def learning_step_torched(
    torch_mps,
    index,
    torch_imgs,
    lr,
    torch_cache,
    inds_dict,
    mask,
    going_right = True,
    update_wrap = lambda site,div: div,
    **kwargs):
    '''
    Compute the updated merged tensor A_{index,index+1}

      UPDATE RULE:  A_{i,i+1} += lr* 2 *( A_{i,i+1}/Z - ( SUM_{i=1}^{m} psi'(v)/psi(v) )/m )
    '''

    # Merge I_k and I_{k+1} in a single rank 4 tensor ('i_{k-1}', 'v_k', 'i_{k+1}', 'v_{k+1}')
    #A = qtn.tensor_contract(mps[index],mps[index+1])
    #Z = qtn.tensor_contract(A,A)
    inds_in = [inds_dict['mps'][index],inds_dict['mps'][index+1]]
    _A, inds_out = torch_contract(tuple(inds_in),torch_mps[index],torch_mps[index+1])
    _Z,_ = torch_contract((inds_out,inds_out),_A,_A)
    A = qtn.Tensor(data = _A.cpu().detach().numpy(),inds = inds_out)

    # Compute the derivative of PSI
    _psi_primed_arr, _inds_out = arr_psi_primed_torched(torch_imgs,torch_cache,index,mask,inds_dict)

    # TODO: Implement submasked version
    _inds_in = [inds_out,_inds_out]
    langsam = torch.unsqueeze(_A,0)[len(mask)*[0]]
    _psi, _ = torch_multicontract(tuple(_inds_in),
                                  langsam,
                                  _psi_primed_arr)

    # Setting the appropriate shape for an element wise quotient
    new_shape = [len(mask),1,1,1]
    if index not in [0,len(torch_mps)-2]:
        new_shape.append(1)
    _psifrac = _psi_primed_arr/torch.reshape(_psi,tuple(new_shape))

    # We want the average
    _psifrac = torch.sum(_psifrac, dim = 0)/len(mask)

    # dNLL = 2A/Z - \frac{2}{|\mathcal{T}|}\sum{\psi'(v)/\psi}
    _dNLL = _A/_Z
    _dNLL = _dNLL.permute(tuple(np.argsort(inds_out))) - _psifrac.permute(tuple(np.argsort(_inds_out)))
    inds_out = tuple(np.sort(inds_out))

    # Go back to quimb for SVD computation on numba
    dNLL = qtn.Tensor(data = _dNLL.cpu().detach().numpy(),inds = inds_out)

    # Release the GPU
    del _dNLL,_psifrac,_psi,_A,_psi_primed_arr,langsam
    torch.cuda.empty_cache()

    # Perform descent
    A = A - _pd.get(lr,'curr_lr',lr)*update_wrap(index, dNLL) # Update A_{i,i+1}

    # Scale
    A = A/A.data.max()

    # Now the tensor A_{i,i+1} must be split in I_k and I_{k+1}.
    # To preserve canonicalization:
    # > if we are merging sliding towards the RIGHT we need to absorb right
    #                                           S  v  D
    #     ->-->--A_{k,k+1}--<--<-   =>   ->-->-->--x--<--<--<-   =>    >-->-->--o--<--<-
    #      |  |    |   |    |  |          |  |  |   |    |  |          |  |  |  |  |  |
    #
    # > if we are merging sliding toward the LEFT we need to absorb left
    #
    if going_right:
        # FYI: split method does apply SVD by default
        # there are variations of svd that can be inspected
        # for a performance boost
        if index == 0:
            SD = A.split(['v'+str(index)], absorb='right', **kwargs)
        else:
            SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='right',**kwargs)
    else:
        if index == 0:
            SD = A.split(['v'+str(index)], absorb='left', **kwargs)
        else:
            SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='left',**kwargs)

    # SD.tensors[0] -> I_{index}
    # SD.tensors[1] -> I_{index+1}
    return SD

def learning_epoch_torched(
    mps,
    imgs,
    torch_mps,
    torch_imgs,
    epochs,
    initial_lr,
    torch_cache,
    inds_dict,
    batch_size = 25,
    update_wrap = lambda site, div: div,
    lr_update = lambda lr: lr,
    **kwargs):
    '''
    Manages the sliding left and right.
    From tensor 1 (the second), apply learning_step() sliding to the right
    At tensor max-2, apply learning_step() sliding to the left back to tensor 1
    '''
    # We expect, however, that the batch size is smaler than the input set
    batch_size = min(len(imgs),batch_size)
    guide = np.arange(len(imgs))
    # Execute the epochs
    cost = []
    lr = copy.copy(initial_lr)
    for epoch in range(epochs):
        print(f'epoch {epoch+1}/{epochs}')
        # [0,1,2,...,780,781,782,782,781,780,...,2,1,0]
        progress = tq.tqdm([i for i in range(0,len(mps.tensors)-1)] + [i for i in range(len(mps.tensors)-2,-1,-1)], leave=True)

        going_right = True
        for index in progress:
            np.random.shuffle(guide)
            mask = guide[:batch_size]
            A = learning_step_torched(
                torch_mps,
                index,
                torch_imgs,
                lr,
                torch_cache,
                inds_dict,
                mask,
                going_right,
                update_wrap,
                **kwargs)
            if index == 0:
                mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(1,0)))
                mps.tensors[index+1].modify(data=A.tensors[1].data)
            else:
                mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(0,2,1)))
                mps.tensors[index+1].modify(data=A.tensors[1].data)

            # update the torched mps
            tens = torch.from_numpy(np.array(mps[index].data,dtype = np.float32))
            if torch.cuda.is_available():
                tens = tens.to('cuda')
            torch_mps[index] = tens
            # Update also the reference dictionary
            inds_dict['mps'][index] = mps[index].inds
            tens = torch.from_numpy(np.array(mps[index+1].data,dtype = np.float32))
            if torch.cuda.is_available():
                tens = tens.to('cuda')
            torch_mps[index+1] = tens
            inds_dict['mps'][index+1] = mps[index+1].inds

            # Update the cache for all images (for all? really?)
            sequential_update_torched(torch_mps,torch_imgs,torch_cache,index,going_right,inds_dict)
            # Place stuff where it belongs:
            #if going_right and index < len(torch_mps) - 2:
            #    torch_mps[index] = torch_mps[index].to('cpu')
            #    torch_mps[index+2] = torch_mps[index+2].to('cuda')
            #    # Current index goes to cpu
            #    # index + 1 stays
            #    # index + 2 is placed in gpu
            #if not going_right and index > 0:
            #    torch_mps[index+1] = torch_mps[index+1].to('cpu')
            #    torch_mps[index-1] = torch_mps[index-1].to('cuda')
            #    # index + 1 goes to cpu
            #    # index stays
            #    # index - 1 is placed in gpu
            #p0 = computepsi(mps,imgs[0])**2
            progress.set_description('Left Index: {}'.format(index))

            if index == len(mps.tensors)-2:
                going_right = False
            torch.cuda.empty_cache()
        #nll = computeNLL(mps,imgs,0)#computeNLL_cached(mps, _imgs, img_cache,0)
        lr = lr_update(lr)
        #print('NLL: {} | Baseline: {}'.format(nll, np.log(len(imgs)) ) )
        #cost.append(nll)
    # cha cha real smooth
    return cost, lr

def learning_step(mps, index, imgs, lr, going_right = True, **kwargs):
    '''
    Compute the updated merged tensor A_{index,index+1}
    
      UPDATE RULE:  A_{i,i+1} += lr* 2 *( A_{i,i+1}/Z - ( SUM_{i=1}^{m} psi'(v)/psi(v) )/m )
      
    '''
    
    # Merge I_k and I_{k+1} in a single rank 4 tensor ('i_{k-1}', 'v_k', 'i_{k+1}', 'v_{k+1}')
    A = (mps.tensors[index] @ mps.tensors[index+1])
    
    # Assumption: The mps is canonized
    Z = A@A
    
    # Computing the second term, summation over
    # the data-dependent terms
    psifrac = 0
    for img in imgs:
        num = computepsiprime(mps,img,index)    # PSI'(v)
        # 'ijkl,ilkj' or 'ijkl,ijkl'?
        # computepsiprime was coded so that the ordering of the indexes is the same
        # as the contraction A = mps.tensors[index] @ mps.tensors[index+1]
        # so it should be the second one
        if index == 0 or index == len(mps.tensors)-2:
            den = np.einsum('ijk,ijk',A.data,num)
        else:
            den = np.einsum('ijkl,ijkl',A.data,num) # PSI(v)
        
        # Theoretically the two computations above can be optimized in a single function
        # because we are contracting the very same tensors for the most part
        
        psifrac = psifrac + num/den
    
    psifrac = psifrac/imgs.shape[0]
    
    # Derivative of the NLL
    dNLL = (A/Z) - psifrac
    
    A = A - lr*dNLL # Update A_{i,i+1}
    A = A/np.sqrt( tneinsum2(A,A).data )
    # Now the tensor A_{i,i+1} must be split in I_k and I_{k+1}.
    # To preserve canonicalization:
    # > if we are merging sliding towards the RIGHT we need to absorb right
    #                                           S  v  D
    #     ->-->--A_{k,k+1}--<--<-   =>   ->-->-->--x--<--<--<-   =>    >-->-->--o--<--<-  
    #      |  |    |   |    |  |          |  |  |   |    |  |          |  |  |  |  |  |
    #
    # > if we are merging sliding toward the LEFT we need to absorb left
    #
    if going_right:
        # FYI: split method does apply SVD by default
        # there are variations of svd that can be inspected
        # for a performance boost
        if index == 0:
            SD = A.split(['v'+str(index)], absorb='right', **kwargs)
        else:
            SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='right',**kwargs)
    else:
        if index == 0:
            SD = A.split(['v'+str(index)], absorb='left', **kwargs)
        else:
            SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='left',**kwargs)
       
    # SD.tensors[0] -> I_{index}
    # SD.tensors[1] -> I_{index+1}
    return SD

def learning_epoch_sgd(mps, imgs, epochs, lr, batch_size = 25,**kwargs):
    '''
    Manages the sliding left and right.
    From tensor 1 (the second), apply learning_step() sliding to the right
    At tensor max-2, apply learning_step() sliding to the left back to tensor 1
    '''

    # We expect, however, that the batch size is smaler than the input set
    batch_size = min(len(imgs),batch_size)
    guide = np.arange(len(imgs))

    # TODO: shouldn't we also consider 0 and 782 here?
    # psi_primed is compatible with this
    # however, computepsiprime only works with:
    # [1,2,...,780,781,780,...,2,1]
    #progress = tq.tqdm([i for i in range(1,len(mps.tensors)-2)] + [i for i in range(len(mps.tensors)-3,0,-1)], leave=True)
    progress = tq.tqdm([i for i in range(0,len(mps.tensors)-1)] + [i for i in range(len(mps.tensors)-2,-1,-1)], leave=True)

    # Firstly we slide right
    going_right = True
    for index in progress:
        np.random.shuffle(guide)
        mask = guide[:batch_size]
        A = learning_step(mps,index,imgs[mask],lr, going_right)
        if index == 0:
            mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(1,0)))
            mps.tensors[index+1].modify(data=A.tensors[1].data)
        else:
            mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(0,2,1)))
            mps.tensors[index+1].modify(data=A.tensors[1].data)

        #p0 = computepsi(mps,imgs[0])**2
        progress.set_description('Left Index: {}'.format(index))

        if index == len(mps.tensors)-2:
            going_right = False

    # cha cha real smooth

def arr_psi_primed_cache(_imgs,img_cache,index):
    if type(img_cache) == daskarr:
        return ext_arr_psi_primed_cache(_imgs,img_cache,index)
    # Extract the cache and free legs
    left_cache = img_cache[:,0,index]
    right_cache = img_cache[:,1,index+1]
    left_imgs = _imgs[:,index]
    right_imgs = _imgs[:,index+1]

    # Contract in parallel
    psi_primed_arr = tneinsum3(left_cache,right_cache,left_imgs,right_imgs)
    return psi_primed_arr

def ext_arr_psi_primed_cache(_imgs,img_cache,index):
    # Extract the cache and free legs
    left_cache = img_cache[:,0,index].compute()
    right_cache = img_cache[:,1,index+1].compute()
    left_imgs = _imgs[:,index]
    right_imgs = _imgs[:,index+1]

    # Contract in parallel
    psi_primed_arr = tneinsum3(left_cache,right_cache,left_imgs,right_imgs)
    return psi_primed_arr

def learning_step_cached(
    mps,
    index,
    _imgs,
    lr,
    img_cache,
    going_right = True,
    update_wrap = lambda site,div: div,
    **kwargs):
    '''
    Compute the updated merged tensor A_{index,index+1}
    
      UPDATE RULE:  A_{i,i+1} += lr* 2 *( A_{i,i+1}/Z - ( SUM_{i=1}^{m} psi'(v)/psi(v) )/m )
    '''

    # Merge I_k and I_{k+1} in a single rank 4 tensor ('i_{k-1}', 'v_k', 'i_{k+1}', 'v_{k+1}')
    A = qtn.tensor_contract(mps[index],mps[index+1])
    Z = qtn.tensor_contract(A,A)

    # Computing the second term, summation over
    # the data-dependent terms
    psi_primed_arr = arr_psi_primed_cache(_imgs,img_cache,index)

    # Generate magical terms
    psi = tneinsum3(np.array(len(_imgs)*[A]),psi_primed_arr)
    psifrac = sum(psi_primed_arr/into_data(psi))
    psifrac = psifrac/len(_imgs)

    # Derivative of the NLL
    dNLL = (A/Z) - psifrac
    
    A = A - _pd.get(lr,'curr_lr',lr)*update_wrap(index, dNLL) # Update A_{i,i+1}
    A = A/A.data.max()#np.sqrt( tneinsum2(A,A).data )
    # Now the tensor A_{i,i+1} must be split in I_k and I_{k+1}.
    # To preserve canonicalization:
    # > if we are merging sliding towards the RIGHT we need to absorb right
    #                                           S  v  D
    #     ->-->--A_{k,k+1}--<--<-   =>   ->-->-->--x--<--<--<-   =>    >-->-->--o--<--<-
    #      |  |    |   |    |  |          |  |  |   |    |  |          |  |  |  |  |  |
    #
    # > if we are merging sliding toward the LEFT we need to absorb left
    #
    if going_right:
        # FYI: split method does apply SVD by default
        # there are variations of svd that can be inspected
        # for a performance boost
        if index == 0:
            SD = A.split(['v'+str(index)], absorb='right', **kwargs)
        else:
            SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='right',**kwargs)
    else:
        if index == 0:
            SD = A.split(['v'+str(index)], absorb='left', **kwargs)
        else:
            SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='left',**kwargs)

    # SD.tensors[0] -> I_{index}
    # SD.tensors[1] -> I_{index+1}
    return SD

def learning_epoch_cached(
    mps,
    val_imgs,
    _imgs,
    epochs,
    initial_lr,
    img_cache,
    batch_size = 25,
    update_wrap = lambda site, div: div,
    lr_update = lambda lr: lr,
    **kwargs):
    '''
    Manages the sliding left and right.
    From tensor 1 (the second), apply learning_step() sliding to the right
    At tensor max-2, apply learning_step() sliding to the left back to tensor 1
    '''
    # We expect, however, that the batch size is smaler than the input set
    batch_size = min(len(_imgs),batch_size)
    guide = np.arange(len(_imgs))
    # Execute the epochs
    cost = []
    lr = copy.copy(initial_lr)
    for epoch in range(epochs):
        print(f'epoch {epoch+1}/{epochs}')
        # [1,2,...,780,781,780,...,2,1]
        #progress = tq.tqdm([i for i in range(0,len(mps.tensors)-1)] + [i for i in range(len(mps.tensors)-3,0,-1)], leave=True)
        #progress = tq.tqdm([i for i in range(1,len(mps.tensors)-2)] + [i for i in range(len(mps.tensors)-3,0,-1)], leave=True)
        progress = tq.tqdm([i for i in range(0,len(mps.tensors)-1)] + [i for i in range(len(mps.tensors)-2,-1,-1)], leave=True)

        going_right = True
        for index in progress:
            np.random.shuffle(guide)
            mask = guide[:batch_size]
            A = learning_step_cached(
                mps,
                index,
                _imgs[mask],
                lr,
                img_cache[mask],
                going_right,
                update_wrap,
                **kwargs)
            if index == 0:
                mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(1,0)))
                mps.tensors[index+1].modify(data=A.tensors[1].data)
            else:
                mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(0,2,1)))
                mps.tensors[index+1].modify(data=A.tensors[1].data)

            # Update the cache for all images (for all? really?)
            sequential_update(mps,_imgs,img_cache,index,going_right)
            #p0 = computepsi(mps,imgs[0])**2
            progress.set_description('Left Index: {}'.format(index))

            if index == len(mps.tensors)-2:
                going_right = False
        nll = computeNLL(mps,val_imgs,0)#computeNLL_cached(mps, _imgs, img_cache,0)
        lr = lr_update(lr)
        print('NLL: {} | Baseline: {}'.format(nll, np.log(len(_imgs)) ) )
        cost.append(nll)
    # cha cha real smooth
    return cost, lr

def cached_stochastic_learning_epoch(mps, val_imgs, _imgs, epochs, lr,img_cache,last_dirs,last_sites,last_epochs,batch_size = 25,**kwargs):
    '''
    Manages the sliding left and right.
    From tensor 1 (the second), apply learning_step() sliding to the right
    At tensor max-2, apply learning_step() sliding to the left back to tensor 1
    '''
    # We expect, however, that the batch size is smaler than the input set
    batch_size = min(len(_imgs),batch_size)
    guide = np.arange(len(_imgs))
    cost = []
    # Execute the epochs
    for epoch in range(epochs):
        print(f'epoch {epoch+1}/{epochs}')
        # [1,2,...,780,781,780,...,2,1]
        #progress = tq.tqdm([i for i in range(0,len(mps.tensors)-1)] + [i for i in range(len(mps.tensors)-3,0,-1)], leave=True)
        #progress = tq.tqdm([i for i in range(1,len(mps.tensors)-2)] + [i for i in range(len(mps.tensors)-3,0,-1)], leave=True)
        progress = tq.tqdm([i for i in range(0,len(mps.tensors)-1)] + [i for i in range(len(mps.tensors)-2,-1,-1)], leave=True)

        going_right = True
        for index in progress:
            np.random.shuffle(guide)
            mask = guide[:batch_size]
            # Update stoch_cache
            stochastic_cache_update(mps,_imgs,img_cache,last_dirs,last_sites,last_epochs,mask,going_right,epoch,index)

            A = learning_step_cached(mps,index,_imgs[mask],lr,img_cache[mask],going_right,**kwargs)
            if index == 0:
                mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(1,0)))
                mps.tensors[index+1].modify(data=A.tensors[1].data)
            else:
                mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(0,2,1)))
                mps.tensors[index+1].modify(data=A.tensors[1].data)

            #p0 = computepsi(mps,imgs[0])**2
            progress.set_description('Left Index: {}'.format(index))

            if index == len(mps.tensors)-2:
                going_right = False

        nll = computeNLL(mps, val_imgs, 0)
        cost.append(nll)
        print('NLL: {} | Baseline: {}'.format(nll, np.log(len(_imgs)) ) )
    return cost
    # cha cha real smooth

def lr_update(lr):
    lr.new_epoch()
    return lr
    
def training_and_probing(
    period_epochs,
    periods,
    mps,
    shape,
    imgs,
    _imgs,
    img_cache,
    batch_size,
    lr,
    lr_update,
    update_wrap,
    val_imgs = [],
    period_samples = 0,
    corrupted_set = None,
    plot = False,
    **kwargs):
    # Initialize the training costs
    train_costs = [computeNLL(mps, imgs,0)]
    #train_costs = []

    # TODO: adapt computeNLL to tneinsum3
    val_costs = []
    if len(val_imgs)>0:
        # Initialize the validation costs
        val_costs.append(computeNLL(mps, val_imgs, 0))


    samples = []
    
    # begin the iteration
    for period in range(periods):
        costs, lr = learning_epoch_cached(mps,imgs,_imgs,period_epochs,lr,img_cache,
                                          lr_update = lr_update,update_wrap = update_wrap,
                                          batch_size = batch_size,**kwargs)
        train_costs.extend(costs)
        if len(val_imgs)>0:
            val_costs.append(computeNLL(mps, val_imgs, 0))
         
        # Save MPS 
        mps_checkpoint(mps, imgs, val_imgs, period, periods, train_costs, val_costs)
        
        # Plot and Save Loss curve
        if plot:
            plot_nll(train_costs,np.log(len(_imgs)),mps,imgs,val_costs, period_epochs)
            plt.show()
        
        # Save Generated Images SVG and NPY
        if period_samples > 0:
            samples.append(generate_and_save(mps, period_samples, period, periods, period_epochs, imgs, shape))
            
    return train_costs, samples
        
    
#   _  _    
#  | || |   
#  | || |_  
#  |__   _| 
#     |_|(_) GENERATION
#######################################################            

def memoize(func):
    ref = {}
    def wrapper(*args):
        result = ref.get(tuple(args),None)
        if result is None:
            result = func(*args)
            ref[tuple(args)] = result
        return result
    return wrapper

@memoize
def zero(ind):
    return qtn.Tensor(data = [0,1],inds=(f'v{ind}',))
@memoize
def one(ind):
    return qtn.Tensor(data = [1,0],inds=(f'v{ind}',))

def generate_samples(mps,N):
    # Iniatlize samples array
    sites = len(mps.tensors)
    samples = np.zeros((N,sites))
    # Assumption: mps is right-canonized
    # Sampling first pixel. Only one probability has to be measured
    ampl = mps[0]@one(0)
    p = ampl@ampl/(mps[0]@mps[0])
    # Generate a random sample for the first pixel of each image:
    rand = np.random.random(N)
    # Which are valid?
    samples[rand <= p,0] = 1# Should this be one() instead? May be useful for reconstruction.

    # Initialize an array of tensors so as to store the half-contractions
    half_contr = np.repeat(ampl,N)
    half_contr[rand>p] = mps[0]@zero(0)

    # iterate over the next sites
    for site in range(1,sites):
        lifechanger = mps[site]@one(site)
        ampl = tneinsum3(half_contr,np.array(N*[lifechanger]))
        p_arr = tneinsum3(ampl,ampl)
        cond_p_arr = into_data(p_arr)/into_data(tneinsum3(half_contr,half_contr))
        rand = np.random.random(N)
        mask = rand <= cond_p_arr
        # The chosen ones
        samples[mask,site] = 1
        # If we are at the final site, we're done
        if site == sites-1:
            # return the samples as they're now complete
            return samples

        # Update those who were accepted
        if np.any(mask):
            half_contr[mask] = ampl[mask]
        # now, proceed with the unwanted
        unmask = np.logical_not(mask)
        # update the rejected ones (better said, the rejected zeros... badum-tss)
        if np.any(unmask):
            extra_term = mps[site]@zero(site)
            half_contr[unmask] = tneinsum3(
                half_contr[unmask],
                np.array(unmask.sum()*[extra_term])
                )

def generate_sample(mps, reconstruct = False):
    '''
    Generate a sample from an MPS.
    0. normalize the mps (probabilities need to be computed)
    1. left canonize the mps (to easily compute the conditional probabilities
    2. Starting from the last pixel (vN):
                                                 __                       __
                                 +---IN         |     +---IN      +---IN    |
                                 |   |       /  |     |   |       |   |     |
                                 | [0,1]    /   |     | [0,1]   + | [1,0]   |
      2.1 P(vN = [0,1]) =        | [0,1]   /    |     | [0,1]     | [1,0]   |
                                 |   |          |     |   |       |   |     |
                                 +---IN         |__   +---+       +---+   __|
                                 
          (The denominator are the sum of probabilities that 
           saturates the state space, the sum of their probabilities,
           being the MPS normalized, is just 1.
           
        2.1.1 Compute the probability of VN being [0,1] then draw a random
              number between 0 and 1, if the random number is less than
              the probabiity computed vN becomes [0,1], else [1,0]
              
              vN then becomes either [0,1] or [1,0] (for the next steps too)
         
         
                                  
      2.k P(vK = [0,1]| v{K+1], ..., vN) = P(vK, v{K+1}, vN) / P(v{K+1}, ..., vN) =
      
             =   +---IK---I{K+1}---...---IN         I{K+1}---...---IN
                 |   |    |              |      /   |              |
                 | [0,1]  v{K+1}         vN    /    v{K+1}         vN
                 | [0,1]  v{K+1}         vN   /     v{K+1}         vN
                 |   |    |              |   /      |              |
                 +---IK---I{K+1}---...---IN         I{K+1}---...---IN
                 
        2.k.1 Compute the probability of vK being [0,1] then draw a random
              number between 0 and 1, if the random number is less than
              the probabiity computed vN becomes [0,1], else [1,0]
              
              vK then becomes either [0,1] or [1,0] (for the next steps too)
              
       .
       .
       .
    '''
    
    
    
    # Canonicalize left
    # We use the property of canonicalization to easily write
    # conditional probabilities
    # By left canonicalizing we will sample from right (784th pixel)
    # to left (1st pixel)
    if not reconstruct:
        mps.left_canonize()
    
    # First pixel
    #   +----In    +
    #   |     |    | half_contr
    #   |     vn   +
    #   |     vn
    #   |     |
    #   +----In
    # To reach efficiency, we gradually contract half_contr with 
    # the other tensors
    # Contract vN to IN
    
    # For the first contraction, we must take into account
    # the mps may not be normalized 
    # for all the other one we have ratios of probabilities 
    # and normalization will not matter
    half_contr = np.einsum('a,ba', [0,1], mps.tensors[-1].data)
    p0 =  half_contr @ half_contr 
    half_contr1 = np.einsum('a,ba', [1,0], mps.tensors[-1].data)
    p1 = half_contr1 @ half_contr1 
    if np.random.rand() < (p0/(p0+p1)):
        generated = deque([0])
    else:
        generated = deque([1])
        # We need to reconstruct half_contr that will be used for the
        # next pixel
        # Contract vN to IN
        half_contr = np.einsum('a,ba', [1,0], mps.tensors[-1].data)
        p =  half_contr @ half_contr
        
    previous_contr = half_contr
        
    for index in range(len(mps.tensors)-2,0,-1):
        # Contract vK to IK
        new_contr = np.einsum('a,bca->bc', [0,1], mps.tensors[index].data)
        # Contract new_contr to the contraction at the previous step
        #   O-- previous_contr
        #   |                  => new_contr -- previous_contr
        #   vK
        new_contr = np.einsum('ab,b', new_contr, previous_contr)
    
        p = (new_contr @ new_contr)/(previous_contr @ previous_contr)
        if np.random.rand() < p:
            generated.appendleft(0)
        else:
            generated.appendleft(1)
            # Contract [1,0] instead of [0,1]
            new_contr = np.einsum('a,bca->bc', [1,0], mps.tensors[index].data)
            new_contr = np.einsum('ab,b', new_contr, previous_contr)
            
            p = (new_contr @ new_contr)/(previous_contr @ previous_contr)
            
        previous_contr = new_contr
    
    # Last pixel
    new_contr = np.einsum('a,ba', [0,1], mps.tensors[0].data)
    new_contr = new_contr @ previous_contr
    
    p = (new_contr**2)/(previous_contr @ previous_contr)
    
    if np.random.rand() < p:
        generated.appendleft(0)
    else:
        generated.appendleft(1)
        
    return generated

def reconstruct_SLOW(mps, corr_img):
    # Copy and normalize psi
    rec_mps = copy.copy(mps)
    rec_mps.normalize()
    
    # Contracting know pixels
    for site, pixel in enumerate(corr_img):
        if pixel == 0:
            if site == 0 or site == len(rec_mps.tensors) - 1:
                data = np.einsum('ab,b',rec_mps[site].data, [0,1] )
                inds = tuple(list(mps.tensors[site].inds)[:-1])
                rec_mps.tensors[site].modify(data=data, inds = inds)
            else:
                data = np.einsum('abc,c',rec_mps[site].data, [0,1] )
                inds = tuple(list(mps.tensors[site].inds)[:-1])
                rec_mps.tensors[site].modify(data=data, inds = inds)
        elif pixel == 1:
            if site == 0 or site == len(rec_mps.tensors) - 1:
                data = np.einsum('ab,b',rec_mps[site].data, [1,0] )
                inds = tuple(list(mps.tensors[site].inds)[:-1])
                rec_mps.tensors[site].modify(data=data, inds = inds)
            else:
                data = np.einsum('abc,c',rec_mps[site].data, [1,0] )
                inds = tuple(list(mps.tensors[site].inds)[:-1])
                rec_mps.tensors[site].modify(data=data, inds = inds)

    reconstructed = copy.copy(corr_img)

    for site, pixel in enumerate(corr_img):
        if pixel == -1:
            den = rec_mps @ rec_mps


            temp_mps = copy.copy(rec_mps)
            if site == 0 or site == len(rec_mps.tensors) - 1:
                data = np.einsum('ab,b',temp_mps[site].data, [0,1] )
                bdata = np.einsum('ab,b',temp_mps[site].data, [1,0] )
            else:
                data = np.einsum('abc,c',temp_mps[site].data, [0,1] )
                bdata = np.einsum('abc,c',temp_mps[site].data, [1,0] )

            inds = tuple(list(rec_mps.tensors[site].inds)[:-1])
            temp_mps.tensors[site].modify(data=data, inds = inds)  

            num = temp_mps @ temp_mps

            p = num/den
            if np.random.rand() < p:
                rec_mps = temp_mps
                reconstructed[site] = 0
            else:
                rec_mps[site].modify(data=bdata, inds=inds)
                reconstructed[site] = 1
                
    return reconstructed

def reconstruct(mps, corr_img):
    
    # Copy the tensor, we need to perform vertical
    # contractions among all know pixels
    rec_mps = copy.copy(mps)
    rec_mps.normalize()
    
    # Contracting know pixels
    corr_img_tn = tens_picture(corr_img)
    
    for site, img_tensor in enumerate(corr_img_tn):
        if img_tensor: # if img_tensor is not None
            contr = tneinsum2(img_tensor, rec_mps[site])
            rec_mps[site].modify(data=contr.data, inds=contr.inds)
    
    first = False # check if we already found an unknown pixel
    upixel = -1
    for site in range(len(rec_mps.tensors)):
        if site == 0:
            ut = rec_mps.tensors[site]
            if 'v0' in rec_mps.tensors[site].inds: # 0 is unknown
                first = True
                upixel = upixel + 1
                ut.modify(tags='U'+str(upixel))
                utn = qtn.Tensor(ut)
                ut = rec_mps.tensors[site+1]
            else:
                ut = tneinsum2(ut, rec_mps.tensors[site+1])
        else:
            if 'v'+str(site) in rec_mps.tensors[site].inds:
                upixel = upixel + 1
                ut.modify(tags='U'+str(upixel))
                if not first:
                    utn = qtn.Tensor(ut)
                else:
                    utn = utn & ut
                first = True
                if site < len(rec_mps.tensors) - 1:
                    ut = rec_mps.tensors[site+1]

            else:
                if site == len(rec_mps.tensors) - 1:
                    finalcontr = tneinsum2(utn.tensors[-1],ut)
                    utn.tensors[-1].modify(data=finalcontr.data, inds = finalcontr.inds)
                else:
                    ut = tneinsum2(ut, rec_mps.tensors[site+1])
                    
    utn = qtn.tensor_1d.TensorNetwork1DFlat(utn.tensors)
    # In order to left canonize i need a class that is a 
    # TensorNetwork1DFlat or MatrixProductState, but
    # I did not manage to transform utn in a MatrixProductState
    
    # The following attributes are needed for leftcanonizing
    utn.cyclic = rec_mps.cyclic
    utn._L = len(utn.tensors)
    utn._site_tag_id = 'U{}'
    
    utn.left_canonize()
    
    # Generate from the unknown pixels network
    reconstruction = generate_sample(utn, reconstruct = True)
    
    rec_img = copy.copy(corr_img)
    rec_img[rec_img == -1] = reconstruction
    
    return rec_img
 
#  ____   
# | ___|  
# |___ \  
#  ___) | 
# |____(_) VISUALIZATION
#######################################################  

def plot_dbonds(mps, savefig=''):
    '''
    Plot the scatter of the bond dimension of every site
    using a colormap to indicate the distance from the center
    
    Supposedly the closer to the margin, the less information is needed
    to be shared
    '''
    
    # side of the image
    L = np.sqrt(len(mps.tensors)).astype(int)
    
    # Extract the distance from the margin
    # for every pixel 
    poss = []
    for pos in range(L*L-1):
        poss.append( [pos//L, pos%L] )  
    poss = np.array(poss)
    poss = poss - L//2
    poss = np.max(np.abs(poss), axis=1)
    
    # Simple cmap for the distance from the margin
    bcolors = [(1, 1, 0), (1, 0, 0)]
    bcmap = LinearSegmentedColormap.from_list('rec', bcolors, N=L//2)
    
    sc = plt.scatter(np.arange(L*L-1), mps.bond_sizes(), c=poss, cmap = bcmap)
    plt.colorbar(sc)
    plt.xlabel('Site')
    plt.ylabel('Bond dimension')
    plt.title('(Right) bond dimension for every pixel')
    
    if savefig != '':
        # save the picture as svg in the location determined by savefig
        plt.savefig(savefig, format='svg')
    plt.show()

def bdims_imshow(mps, shape, savefig=''):
    heat = np.append(np.array(mps.bond_sizes()),0).reshape(shape)
    hm = plt.imshow(heat)
    plt.colorbar(hm)
    
    plt.title('(Right) bond dimension for every pixel')
    
    if savefig != '':
        # save the picture as svg in the location determined by savefig
        plt.savefig(savefig, format='svg')
    plt.show()
        
#  __    
# / /_   
#| '_ \  
#| (_) | 
# \___(_) OTHER
#######################################################

def bars_n_stripes(N_samples, dim = 4):
    if N_samples > 30:
        N_samples = 30
        
    samples = []
    _ = 0
    while _ < N_samples:
        sample = np.zeros((dim,dim))
        guide = np.random.random(dim+1)
        if guide[0]<=0.5:
            sample[guide[1:]<=0.5,:] = 1
        else:
            sample[:,guide[1:]<=0.5] = 1
        
        # If the image just generated is already present
        # in the list, we need to rerun this cycle
        if not any((sample == x).all() for x in samples):
            samples.append(sample)
            _ = _ + 1
            
    return samples

def save_mps_sets(mps, train_set, foldname, test_set = []):
    # If folder does not exists
    if not os.path.exists('./'+foldname):
        # Make the folder
        os.makedirs('./'+foldname)
        
    # Save the mps
    quimb.utils.save_to_disk(mps, './'+foldname+'/mps')
    
    # Save the images
    np.save('./'+foldname+'/train_set.npy', train_set)
    
    if len(test_set) > 0:
        np.save('./'+foldname+'/test_set.npy', test_set)
        
def load_mps_sets(foldname):
    # Load the mps
    mps = quimb.utils.load_from_disk('./'+foldname+'/mps')
    
    # Load the images
    train_set = np.load('./'+foldname+'/train_set.npy')
    
    if os.path.isfile('./'+foldname+'/test_set.npy'):
        test_set =  np.load('./'+foldname+'/test_set.npy')
    else: test_set = None
    
    return mps, train_set, test_set

def meanpool2d(npmnist, shape, grayscale_threshold = 0.3):
    '''
    Apply a meanpool convolution of an array of images (flattened)
    meanpool has kernel size 2x2
    '''
    ds_imgs = []
    for img in npmnist:
        ds_img = []
        for col in range(0,shape[0],2):
            for row in range(0,shape[1],2):
                pixel = np.mean([img.reshape(shape)[col,row], img.reshape(shape)[col,row+1],
                                 img.reshape(shape)[col+1,row], img.reshape(shape)[col+1,row+1]])
                
                ds_img.append(pixel)

        ds_imgs.append(np.array(ds_img).reshape((shape[0]//2)*(shape[1]//2)) )
        
    ds_imgs = np.array(ds_imgs)
    
    ds_imgs[ds_imgs > grayscale_threshold] = 1
    ds_imgs[ds_imgs <= grayscale_threshold] = 0
    
    return ds_imgs

def mps_checkpoint(mps, imgs, val_imgs, period, periods, train_cost, val_cost, path = './'):
    # Save the mps
    oldfilename = str(period-1)+'I'+str(periods-1)
    filename = str(period)+'I'+str(periods-1)
    foldname = 'T'+str(len(imgs))+'_L'+str(len(mps.tensors))
    # If folder does not exists
    if not os.path.exists(path+foldname):
        # Make the folder
        os.makedirs(path+foldname)

    if os.path.isfile(path+foldname+'/'+oldfilename+'.mps'):
        os.remove(path+foldname+'/'+oldfilename+'.mps')

    quimb.utils.save_to_disk(mps, path+foldname+'/'+filename+'.mps')

    # Save training and val loss
    np.save(path+foldname+'/trainloss', train_cost)
    np.save(path+foldname+'/valloss', val_cost)

    # Save img and val_img
    if period == 0:
        np.save(path+foldname+'/train_set.npy', imgs)
        np.save(path+foldname+'/val_set.npy', val_imgs)

    # Where are we placing stuff now?
    return path + foldname + '/'

def generate_and_save(mps, period_samples, period, periods, period_epochs, imgs, shape, path = './'):
    # If images are more than one, we display using subplots
    if period_samples > 1:
        # Generate images
        gen_imgs = generate_samples(mps,period_samples)
        # Create the subfolder for gen_imgs if it does not exists
        if not os.path.exists(path+'/gen_imgs'):
            os.mkdir(path+'/gen_imgs')

        fig, ax = plt.subplots(1, period_samples, figsize=(2*period_samples,2))
        for i in range(period_samples):
            ax[i].imshow(1-gen_imgs[i].reshape(shape), cmap='gray')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        fig.suptitle('Generated images: {}/{}'.format(period_epochs*(period+1),period_epochs*periods))
        plt.savefig(path+'/gen_imgs/'+str(period)+'.svg', format='svg')
    
    # if period_samples == 1, we display using plot_img function    
    else:
        # Generate images
        gen_imgs = generate_sample(mps)
        # Create the subfolder for gen_imgs if it does not exists
        if not os.path.exists(path+'/gen_imgs'):
            os.mkdir(path+'/gen_imgs')

        plot_img(gen_imgs, shape, border = True, 
                 title = 'Generated image: {}/{}'.format(period_epochs*(period+1),period_epochs*periods),
                 savefig = path+'/gen_imgs/'+str(period)+'.svg' )
    plt.show()
    
    # Save the npy of the current generated images
    np.save(path +'/gen_imgs/'+str(period)+'.npy', gen_imgs)
    
    return gen_imgs

def plot_nll(nlls,baseline,val_nlls,period_epochs = 1, path = './'):
    plt.plot(range(len(nlls)),nlls, label='training set')
    plt.title('Negative log-likelihood')
    plt.axhline(baseline,color = 'r', linestyle= 'dashed', label='baseline')
    if len(val_nlls) > 0:
        plt.plot(range(0,len(nlls),period_epochs),val_nlls, label='test set')
    plt.legend()
    step = np.round(len(nlls)/10).astype(int)
    if step == 0: step = 1
    plt.xticks(range(0,len(nlls),step)) # Ticks only show integers (epochs)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}$')
    plt.grid()
    if not os.path.exists(path):
            os.mkdir(path)
    plt.savefig(path+'/loss.svg', format='svg')

