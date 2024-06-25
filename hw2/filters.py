import numpy as np
import torch


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image = np.pad(image, Hk // 2)

    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for l in range(Wk):
                    out[i, j] += image[i + k, j + l] * kernel[Hk - 1 - k, Wk - 1 - l]

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros_like(image)

    if pad_width > pad_height:
        out = np.pad(image, pad_height)
        h, w = out.shape
        sup = pad_width - pad_height
        out = np.insert(out, [0]*sup, np.zeros(h), axis=1)
        out = np.insert(out, [-1]*sup, np.zeros(h), axis=1)
    else:
        out = np.pad(image, pad_width)
        h, w = out.shape
        sup = pad_height - pad_width
        out = np.insert(out, [0]*sup, np.zeros(w), axis=0)
        out = np.insert(out, [-1]*sup, np.zeros(w), axis=0)

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    img_pad = zero_pad(image, Hk // 2, Wk // 2)
    kernel = np.flip(kernel)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(img_pad[i:i + Hk, j:j + Wk] * kernel)

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    kernel = np.flip(kernel)
    torch_kernel = torch.tensor(kernel.copy(), dtype=torch.float32)
    torch_kernel = torch_kernel.reshape(1, 1, Hk, Wk)

    conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(Hk, Wk), padding='same', bias=False)
    conv.weight = torch.nn.Parameter(torch_kernel)

    torch_image = torch.tensor(image, dtype=torch.float32)
    torch_image = torch_image.reshape(1, 1, Hi, Wi)
    out = conv(torch_image).detach().numpy().reshape((Hi, Wi))

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    tg = g.astype(np.float64)
    ig = f.astype(np.float64)

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    img_pad = zero_pad(ig, Hk // 2, Wk // 2)
    sum_sq_tg = np.sum(tg ** 2)

    for i in range(Hi):
        for j in range(Wi):
            img_slice = img_pad[i:i + Hk, j:j + Wk]
            koeff = np.sqrt(sum_sq_tg * np.sum(img_slice ** 2))
            out[i, j] = np.sum(img_slice * tg) / koeff

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    temp = g - np.mean(g)
    out = cross_correlation(f, temp)

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    tg = g.astype(np.float64)
    ig = f.astype(np.float64)

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    img_pad = zero_pad(ig, Hk // 2, Wk // 2)

    sigma = np.std(tg)
    mid = np.mean(tg)
    norm_tg = (tg - mid) / sigma
    sum_sq_tg = np.sum(tg ** 2)

    for i in range(Hi):
        for j in range(Wi):
            img_slice = img_pad[i:i + Hk, j:j + Wk]
            koeff = np.sqrt(sum_sq_tg * np.sum(img_slice ** 2))
            out[i, j] = np.sum(((img_slice - np.mean(img_slice)) / np.std(img_slice)) * norm_tg) / koeff

    return out
