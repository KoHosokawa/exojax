import numpy as np
import tqdm


def uniqidx(input_array):
    """ compute indices based on uniq values of the input M-dimensional array.                                                   
                                                                                                                        
    Args:                                                                                                               
        input_array: input array (N,M), will use unique M-dim vectors                                                             
                                                                                                                        
    Returns:                                                                                                            
        unique index, unique value                                                                                      
                                                                                                                        
    Examples:                                                                                                           
                                                                                                                        
        >>> a=np.array([[4,1],[7,1],[7,2],[7,1],[8,0],[4,1]])                                                           
        >>> uidx, uval=uniqidx(a) #->[0,1,2,1,3,0], [[4,1],[7,1],[7,2],[8,0]]                                        
                                                                                                                        
    """
    N, _ = np.shape(input_array)
    uniqvals = np.unique(input_array, axis=0)
    uidx = np.zeros(N, dtype=int)
    uidx_p = np.where(input_array == uniqvals[0], True, False)
    uidx[np.array(np.prod(uidx_p, axis=1), dtype=bool)] = 0
    for i, uv in enumerate(tqdm.tqdm(uniqvals[1:], desc="uniqidx")):
        uidx_p = np.where(input_array == uv, True, False)
        uidx[np.array(np.prod(uidx_p, axis=1), dtype=bool)] = i + 1
    return uidx, uniqvals


def uniqidx_plus_one(index_array):
    """ compute indices based on uniq values of the input index array and input index + one vector  
                                                                                                                        
    Args:                                                                                                               
        index_array: input index array (N,M), will use unique M-dim vectors                                                             
                                                                                                                        
    Returns:                                                                                                            
        unique index, unique value                                                                                      
                                                                                                                        
    Examples:                                                                                                           
                                                                                                                        
        >>> a=np.array([[4,1],[7,1],[7,2],[7,1],[8,0],[4,1]])                                                           
        >>> uidx, uval=uniqidx_2D(a) #->[0,1,2,1,3,0], [[4,1],[7,1],[7,2],[8,0]]                                        
                                                                                                                        
    """
    uidx, uniqval = uniqidx(index_array)
    uniqval_new=np.copy(uniqval)
    Nuidx = np.max(uidx)
    neighbor_indices=np.zeros((Nuidx,3))
    for i in range(0, Nuidx):
        neighbor_indices[i,0], uniqval_new = find_or_add_index(uniqval[i,:]+np.array([1,0]), uniqval_new)
        neighbor_indices[i,1], uniqval_new = find_or_add_index(uniqval[i,:]+np.array([0,1]), uniqval_new)
        neighbor_indices[i,2], uniqval_new = find_or_add_index(uniqval[i,:]+np.array([1,1]), uniqval_new)
        
    return False

def find_or_add_index(new_index, index_array):
    """find a position of a new index in index_array, if not exisited add the new index in index_array
    
    Args: 
        new_index: new index investigated
        index_array: index array
   
    Returns:
        position, index_array updated
        
    """
    uidx_p = np.where(index_array == new_index, True, False)
    mask = np.array(np.prod(uidx_p, axis=1), dtype=bool)
    ni = np.where(mask == True)[0]
    if len(ni) == 0:
        index_array = np.vstack([index_array,new_index])
        return len(index_array) - 1, index_array
    else:
        return ni[0], index_array

if __name__ == "__main__":
    a = np.array([[4, 1], [7, 1], [7, 2], [7, 1], [8, 0], [5, 1]])
    uniqidx_plus_one(a)

