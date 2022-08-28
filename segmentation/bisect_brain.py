import numpy as np
import skimage

def lingress (X, Y, W=None):
    '''Vectorised linear regression
    X, Y, W are 2D arrays in shape of (batch, dimension)
    W is the binary matrix to exclude some samples
    '''
    if W is None: W = np.ones(X.shape)
    mu_x = (W*X).sum(axis=1, keepdims=True)/W.sum(axis=1, keepdims=True)
    mu_y = (W*Y).sum(axis=1, keepdims=True)/W.sum(axis=1, keepdims=True)
    var_x = np.sum (W*(X - mu_x)**2, axis=1)
    b1 = np.sum (W*(X-mu_x)*(Y-mu_y), axis=1)/var_x
    b0 = mu_y.squeeze() - mu_x.squeeze()*b1
    return b0, b1

def nonzero_min (x, axis=0, replace=1e10):
    ''' 
    Find the minimum that is not 0 along an axis 
    Args:
        `replace`: replace all zeros with a very large number, to prevent it
        from being selected as minimum
    '''
    min_x = x.copy()
    min_x[min_x==0]= replace
    min_x= min_x.min(axis=axis)
    min_x [min_x==replace] = 0
    return min_x

def bisect (img, thres=0, substance=0.3):
    '''
    Bisect a structure along its y axis (across the x axis)
    This forms a 2D plane to bisect a 3D structure.
    To bisect a 2D structure, just set the depth to 1
    Args:
        `img`: [height, width, depth]
        `thres`: threshold to binarise the image
        `substance`: percentage of a line must not be background before the
        line is deemed to be passing through brain substance
    '''
    # build a coordinate grid
    H, W, D = img.shape
    grid2d = np.stack ([np.arange (H)]*W, axis=-1)
    grid3d = np.stack ([grid2d]*D, axis=-1)

    img_mask = (img>thres).astype(int)
    coord = grid3d*img_mask # masked coordinates

    # midpoint along each y coordinate is defined as the center of two edges of
    # the brain
    min_coord = nonzero_min (coord, axis=0)
    mean_coord = (min_coord+coord.max(axis=0))/2

    # exclude the lines that do not cross brain substance
    weight = img_mask.mean(axis=0)>substance
    grid2d_y = np.stack ([np.arange (W)]*D, axis=-1)
    b0, b1 = lingress (grid2d_y.T, mean_coord.T, weight.T)

    # exclude the slopes and intercepts that were not calculated using
    # a suffcient number of samples
    b_weight = weight.mean(axis=0) > substance
    b1 *= b_weight
    b0 *= b_weight
    x_coord = grid2d_y*b1.reshape([1,-1]) + b0.reshape([1,-1])
    return x_coord

def line2img (img_shape, centerline):
    '''
    Convert line coordinate into an image for display
    Examples:
    >>> import matplotlib.pyplot as plt
    >>> import view.visual_select as VS
    >>> import data_raw.CTname as CTN
    >>> img = CTN.load_nib ('data/tmp_images/MGH1.nii.gz')
    >>> mask = (img>0.5).sum(axis=-1)>=1
    >>> centerline = bisect (mask)
    >>> cenimg = line2img (mask.shape, centerline)
    >>> VS.show_levels (mask, overlay=cenimg.astype(int))
    >>> plt.show()
    '''
    img = np.zeros (img_shape)
    H, W, D = img_shape
    grid2d_y = np.stack ([np.arange (W)]*D, axis=-1)
    grid2d_z = np.stack ([np.arange (D)]*W, axis=-1)
    center_cat = np.stack ([centerline, grid2d_y, grid2d_z.T], axis=-1)
    center_cat = center_cat.reshape ([-1, 3])
    center_cat = center_cat [~np.isnan(center_cat[:,0])]
    center_cat = (np.round (center_cat)).astype(int)
    img [center_cat[:,0], center_cat[:,1], center_cat[:,2]] = 1
    return skimage.filters.gaussian(img) >0

def quant_tissue (img, mask, window=[-100,100]):
    '''
    Obtain the volume and mean intensity of left and right hemisphere
    This function assumes that the rostral-caudal axis of the brain is along
    the y axis (axis=1). Please permute your image axis accordingly if needed. 
    This function also assumes that the rostral aspect is to the right side
    (extreme side of the y axis). Please flip the brain along the y axis if
    this is not the case.
    Args:
        `img`: 3d array of [height, width, depth]
        `mask`: 4d array of [height, width, depth, 3]
    '''
    centerline = bisect (img>0)
    img_win = img.clip (window[0], window[1])[...,np.newaxis]
    mask = (mask >0.5).astype(int)

    H, W, D, _ = mask.shape
    grid2d = np.stack ([np.arange (H)]*W, axis=-1)
    grid3d = np.stack ([grid2d]*D, axis=-1)

    right_hemi = (grid3d < centerline[np.newaxis]).astype(int)
    left_hemi = (grid3d > centerline[np.newaxis]).astype(int)
    keep_slices = (np.nansum(centerline, axis=0) !=0).astype(int)
    keep_slices = keep_slices.reshape ([1,1,D,1])

    right_tissue = mask*right_hemi[...,np.newaxis]*keep_slices
    left_tissue = mask*left_hemi[...,np.newaxis]*keep_slices
    right_vol = right_tissue.sum(axis=(0,1,2))
    left_vol = left_tissue.sum(axis=(0,1,2))
    right_HU = (right_tissue*img_win).sum((0,1,2))
    left_HU = (left_tissue*img_win).sum((0,1,2))
    return np.array (list(right_vol)+list(left_vol)+list(right_HU/right_vol)+
            list(left_HU/left_vol))
