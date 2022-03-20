#################
#### IMPORTS ####
#################

# Arrays
import numpy as np
import cytoolz

# Deep Learning stuff
import torch
import torchvision
import torchvision.transforms as transforms

# Images display and plots
import matplotlib.pyplot as plt

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

def plot_img(img_flat, flip_color = True, savefig = ''):
    '''
    Display the image from the flattened form
    '''
    # If the image is corrupted for partial reconstruction (pixels are set to -1)
    if -1 in img_flat:
        img_flat = np.copy(img_flat)
        img_flat[img_flat == -1] = 0
    
    # Background white, strokes black
    if flip_color:
        plt.imshow(1-np.reshape(img_flat,(28,28)), cmap='gray')
    # Background black, strokes white
    else:
        plt.imshow(np.reshape(img_flat,(28,28)), cmap='gray')
        
    plt.axis('off')
    
    if savefig != '':
        # save the picture as svg in the location determined by savefig
        plt.savefig(savefig, format='svg')
        plt.show()
        
def partial_removal_img(mnistimg, fraction = .5, axis = 0):
    '''
    Corrupt (with -1 values) a portion of an input image (from the test set)
    to test if the algorithm can reconstruct it
    '''
    # Check type:
    if [type(mnistimg), type(fraction), type(axis)] != [np.ndarray, float, int]:
        raise TypeError('Input types not valid')
    
    # Check the shape of input image
    if (mnistimg.shape[0] != 784):
        raise TypeError('Input image shape does not match, need (784,)')
    
    # Axis can be either 0 (rowise deletion) or 1 (columnwise deletion)
    if not(axis in [0,1]):
        raise ValueError('Invalid axis [0,1]')
    
    # Fraction must be from 0 to 1
    if (fraction < 0 or fraction > 1):
        raise ValueError('Invalid value for fraction variable (in interval [0,1])')
        
    mnistimg_corr = np.copy(mnistimg)
    mnistimg_corr = np.reshape(mnistimg_corr, (28,28))
    
    if axis == 0:
        mnistimg_corr[int(28*(1-fraction)):,:] = -1
    else:
        mnistimg_corr[:,int(28*(1-fraction)):] = -1
        
    mnistimg_corr = np.reshape(mnistimg_corr, (784,))
    
    return mnistimg_corr

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

def initialize_mps(Ldim = 28*28, bdim = 30, canonicalize = 1):
    '''
    Initialize the MPS tensor network
    1. Create the MPS TN
    2. Canonicalization
    3. Renaming indexes
    '''
    # Create a simple MPS network randomly initialized
    mps = qtn.MPS_rand_state(L=Ldim, bond_dim=bdim)
    
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
    mps = mps.reindex({mps.tensors[tensor].inds[0]: 'i'+str(tensor), 
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

stater = lambda x: [0,1] if x == 0 else [1,0]
def tens_picture(picture):
    '''Converts an array of bits into a list of tensors compatible with a tensor network.'''
    tens = [qtn.Tensor(stater(n),inds=(f'v{i}',)) for i, n in enumerate(picture)]
    return tens

def left_right_cache(_imgs):
    # Cache
    # For each image, we compute the left vector for each site, (?) as well as the right vector(?)
    # update it each time a site is updated
    img_cache = []
    for img in _imgs:
        # Instead of contracting, just take mps[0][:,0] or mps[0][:,1]
        curr_l = qtn.Tensor()#mps[0]@img[0]
        curr_r = qtn.Tensor()
        left_cache = [curr_l]
        right_cache = [curr_r]
        for site in range(len(img)-1):
            contr_l = mps[site]@curr_l
            curr_l = contr_l@img[site]
            left_cache.append(curr_l)
            contr_r = mps[-(site+1)]@curr_r
            curr_r = contr_r@img[-(site+1)]
            right_cache.append(curr_r)
        # reversing the right cache for site indexing consistency
        right_cache.reverse()
        img_cache.append((left_cache,right_cache))
    return img_cache

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

def computeNLL(mps, imgs):
    '''
    Computes the Negative Log Likelihood of a Tensor Network (mps)
    over a set of images (imgs)
    
     > NLL = -(1/|T|) * SUM_{v\in T} ( ln P(v) ) = -(1/|T|) * SUM_{v\in T} ( ln psi(v)**2 )
           = -(2/|T|) * SUM_{v\in T} ( ln |psi(v)| )
    '''
    
    lnsum = 0
    for img in imgs:
        lnsum = lnsum + np.log( abs(computepsi(mps,img)) )
        
    return - 2 * lnsum / imgs.shape[0]

#   _____  
#  |___ /  
#    |_ \  
#   ___) | 
#  |____(_) LEARNING FUNCTIONS
#######################################################        

def learning_step(mps, index, imgs, lr, going_right = True):
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
        den = np.einsum('ijkl,ijkl',A.data,num) # PSI(v)
        
        # Theoretically the two computations above can be optimized in a single function
        # because we are contracting the very same tensors for the most part
        
        psifrac = psifrac + num/den
    
    psifrac = psifrac/imgs.shape[0]
    
    # Derivative of the NLL
    dNLL = (A/Z) - psifrac
    
    A = A + lr*dNLL # Update A_{i,i+1}
    
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
        SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='right')
    else:
        SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='left')
       
    # SD.tensors[0] -> I_{index}
    # SD.tensors[1] -> I_{index+1}
    return SD

def learning_step_numpy(mps, index, imgs, lr, going_right = True):
    '''
    DOES NOT WORK
    
    Compute the updated merged tensor A_{index,index+1}
    
      UPDATE RULE:  A_{i,i+1} += lr* 2 *( A_{i,i+1}/Z - ( SUM_{i=1}^{m} psi'(v)/psi(v) )/m )
      
    '''
    
    # Merge I_k and I_{k+1} in a single rank 4 tensor ('i_{k-1}', 'v_k', 'i_{k+1}', 'v_{k+1}')
    # OLD: A = (mps.tensors[index] @ mps.tensors[index+1])
    # '@' may be too slow
    if index == 0:
        A = np.einsum('ij,iab->jab',mps.tensors[index].data,mps.tensors[index+1].data)
        # Assumption: The mps is canonized
        Z = np.einsum('jab,jab',A,A)
    elif index == (len(mps.tensors)-2):
        A = np.einsum('ijk,ja->ika',mps.tensors[index].data,mps.tensors[index+1].data)
        # Assumption: The mps is canonized
        Z = np.einsum('ika,ika',A,A)
    else:
        A = np.einsum('ijk,jab->ikab',mps.tensors[index].data,mps.tensors[index+1].data)
        # Assumption: The mps is canonized
        Z = np.einsum('ikab,ikab',A,A)
    
    # Computing the second term, summation over
    # the data-dependent terms
    psifrac = 0
    for img in imgs:
        num = computepsiprime(mps,img,index)    # PSI'(v)
        # 'ijkl,ilkj' or 'ijkl,ijkl'?
        # computepsiprime was coded so that the ordering of the indexes is the same
        # as the contraction A = mps.tensors[index] @ mps.tensors[index+1]
        # so it should be the second one    
        den = np.einsum('ijkl,ijkl',A.data,num) # PSI(v)
        
        # Theoretically the two computations above can be optimized in a single function
        # because we are contracting the very same tensors for the most part
        
        psifrac = psifrac + num/den
    
    psifrac = psifrac/imgs.shape[0]
    
    # Derivative of the NLL
    dNLL = (A/Z) - psifrac
    
    A = A + lr*dNLL # Update A_{i,i+1}
    
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
        SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='right')
    else:
        SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='left')
       
    # SD.tensors[0] -> I_{index}
    # SD.tensors[1] -> I_{index+1}
    return SD

def learning_epoch_sgd(mps, imgs, epochs, lr, batch_size = 25):
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
    progress = tq.tqdm([i for i in range(1,len(mps.tensors)-2)] + [i for i in range(len(mps.tensors)-3,0,-1)], leave=True)

    # Firstly we slide right
    going_right = True
    for index in progress:
        np.random.shuffle(guide)
        mask = guide[:batch_size]
        A = learning_step(mps,index,imgs[mask],lr, going_right)

        mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(0,2,1)))
        mps.tensors[index+1].modify(data=A.tensors[1].data)

        #p0 = computepsi(mps,imgs[0])**2
        progress.set_description('Left Index: {}'.format(index))

        if index == len(mps.tensors)-3:
            going_right = False

    # cha cha real smooth

def learning_step_cached(mps, index, _imgs, lr, img_cache, going_right = True):
    '''
    Compute the updated merged tensor A_{index,index+1}
    
      UPDATE RULE:  A_{i,i+1} += lr* 2 *( A_{i,i+1}/Z - ( SUM_{i=1}^{m} psi'(v)/psi(v) )/m )
    '''

    # Merge I_k and I_{k+1} in a single rank 4 tensor ('i_{k-1}', 'v_k', 'i_{k+1}', 'v_{k+1}')
    A = (mps.tensors[index]@mps.tensors[index+1])
    Z = A@A

    # Computing the second term, summation over
    # the data-dependent terms
    psifrac = 0
    for _img,cache in zip(_imgs,img_cache):
        L, R = cache
        num = L[index]@R[index+1]@_img[index]@_img[index+1]
        den = num@A


        # Theoretically the two computations above can be optimized in a single function
        # because we are contracting the very same tensors for the most part

        psifrac = psifrac + num/den

    psifrac = psifrac/len(_imgs)

    # Derivative of the NLL
    dNLL = (A/Z) - psifrac

    A = A + lr*dNLL # Update A_{i,i+1}

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
        SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='right')
    else:
        SD = A.split(['i'+str(index-1),'v'+str(index)], absorb='left')

    # SD.tensors[0] -> I_{index}
    # SD.tensors[1] -> I_{index+1}
    return SD

def learning_epoch_cached(mps, _imgs, epochs, lr,img_cache):
    '''
    Manages the sliding left and right.
    From tensor 1 (the second), apply learning_step() sliding to the right
    At tensor max-2, apply learning_step() sliding to the left back to tensor 1
    '''
    for epoch in range(epochs):
        print(f'epoch {epoch+1}/{epochs}')
        # [1,2,...,780,781,780,...,2,1]
        progress = tq.tqdm([i for i in range(1,len(mps.tensors)-2)] + [i for i in range(len(mps.tensors)-3,0,-1)], leave=True)

        # Firstly we slide right
        going_right = True
        for index in progress:
            A = learning_step_cached(mps,index,_imgs,lr,img_cache,going_right)

            mps.tensors[index].modify(data=np.transpose(A.tensors[0].data,(0,2,1)))
            mps.tensors[index+1].modify(data=A.tensors[1].data)

            # Update the cache for all images (for all? really?)
            for i,cache in enumerate(img_cache):
                if going_right:
                    # updating left
                    img_cache[i][0][index+1] = mps[index]@cache[0][index]@_imgs[i][index]
                else:
                    # updating right
                    img_cache[i][1][index] = mps[index+1]@cache[1][index+1]@_imgs[i][index+1]
            #p0 = computepsi(mps,imgs[0])**2
            progress.set_description('Left Index: {}'.format(index))

            if index == len(mps.tensors)-3:
                going_right = False

    # cha cha real smooth

#   _  _    
#  | || |   
#  | || |_  
#  |__   _| 
#     |_|(_) GENERATION
#######################################################            

def generate_sample(mps):
    
    mps = mps / mps.norm()
    
    # It is clear that this can be easily performed if we 
    # have gauged all the tensors except A_N to be left canonical 
    mps.left_canonize()
    
    # First pixel
    #   +----In    +
    #   |     |    | half_contr
    #   |     vn   +
    #   |     vn
    #   |     |
    #   +----In
    half_contr = np.einsum('a,ba', [0,1], mps.tensors[-1].data)
    p =  half_contr @ half_contr
    
    if np.random.rand() < p:
        generated = deque([0])
    else:
        generated = deque([1])
        half_contr = np.einsum('a,ba', [1,0], mps.tensors[-1].data)
        p =  half_contr @ half_contr
        
    previous_contr = half_contr
        
    for index in range(len(mps.tensors)-2,0,-1):
        new_contr = np.einsum('a,bca->bc', [0,1], mps.tensors[index].data)
        new_contr = np.einsum('ab,b', new_contr, previous_contr)
    
        p = (new_contr @ new_contr)/(previous_contr @ previous_contr)
        
        if np.random.rand() < p:
            generated.appendleft(0)
        else:
            generated.appendleft(1)
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
