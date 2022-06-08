import torch
from torch import Tensor
# typing
import collections
from typing import Union, Any, Sequence, Dict, Callable

def map_tensors(
    tensors:Union[Dict[Any, Tensor], Sequence[Tensor], Tensor], 
    fn:Callable[Tensor, Any]
) -> Union[Dict[Any, Tensor], Sequence[Tensor], Tensor]:
    
    # trivial case
    if isinstance(tensors, Tensor):
        return fn(tensors)

    # handler sequences
    elif isinstance(tensors, collections.abc.Sequence):
        moved_list = [map_tensors(t, fn=fn) for t in tensors]
        # try to convert it to the same type
        try:
            return type(tensors)(moved_list)
        except TypeError:
            # might not support initialization from interator
            return moved_list
    
    # handle mappings
    elif isinstance(tensors, collections.abc.Mapping):
        moved_dict = {key: map_tensors(t, fn=fn) for key, t in tensors.items()}
        # try to convert it to the same type
        try:
            return type(tensors)(moved_dict)
        except TypeError:
            # might not support initialization from dict
            return moved_dict
    
    # fallback
    return tensors

def move_to_device(
    tensors:Union[Dict[Any, Tensor], Sequence[Tensor], Tensor], 
    device:torch.device
) -> Union[Dict[Any, Tensor], Sequence[Tensor], Tensor]:
    # move all tensors to the given device
    return map_tensors(tensors, fn=lambda t: t.to(device))

def concat_tensors(batch):
    """ torch's `default_collate` function but altered to concatenate instead of stack """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError()

            return concat_tensors([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: concat_tensors([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: concat_tensors([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(concate_tensors(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        try:
            return elem_type([concat_tensors(samples) for samples in transposed])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [concat_tensors(samples) for samples in transposed]

    raise TypeError()

