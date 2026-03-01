import numpy as np


def split_data(
    x_train,
    y_train,
    mp_size,
    dp_size,
    rank,
):
    """The function for splitting the dataset uniformly across data parallel groups

    Parameters
    ----------
        x_train : np.ndarray float32
            the input feature of MNIST dataset in numpy array of shape (data_num, feature_dim)

        y_train : np.ndarray int32
            the label of MNIST dataset in numpy array of shape (data_num,)

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        rank : int
            the corresponding rank of the process

    Returns
    -------
        split_x_train : np.ndarray float32
            the split input feature of MNIST dataset in numpy array of shape (data_num/dp_size, feature_dim)

        split_y_train : np.ndarray int32
            the split label of MNIST dataset in numpy array of shape (data_num/dp_size, )

    Note
    ----
        please split the data uniformly across data parallel groups and
        do not shuffle the index as we will shuffle them later
    """
    # Try to get the correct start_idx and end_idx from dp_size, mp_size and rank and return
    # the corresponding data
    
    # assuming index row major order in dp_size x mp_size (row major order increasing along mp_size)
    
    n,_ = x_train.shape
    
    # assume n % dp_size == 0
    chunk_size = n // dp_size

    # 0-mp_size-1 -> idx=0
    idx = rank // mp_size

    x_slice = x_train[idx*chunk_size:(idx+1)*chunk_size, :]
    y_slice = y_train[idx*chunk_size:(idx+1)*chunk_size]
    return x_slice, y_slice
