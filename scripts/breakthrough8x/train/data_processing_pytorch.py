import logging
import os

import numpy as np

import torch
import torch.nn.functional

import concurrent.futures
import modelconfigs

def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: int,
    value_sampling_proportion: float,
    model_config: modelconfigs.ModelConfig,
):
    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)

    #create loading file thread
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None

    def load_npz_file(npz_file):
        with np.load(npz_file) as npz:
            binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
            globalInputNC = npz["globalInputNC"]
            policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
            globalTargetsNC = npz["globalTargetsNC"]
            scoreDistrN = npz["scoreDistrN"].astype(np.float32)
            valueTargetsNCHW = npz["valueTargetsNCHW"].astype(np.float32)
        del npz

        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
        )).astype(np.float32)

        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        return [binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW]

    #read the first file
    future = executor.submit(load_npz_file, npz_files[0])
    npz_files.append("")
    npz_files=npz_files[1:]
    for npz_file in npz_files:
        binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW = future.result()

        if npz_file != "":
            future = executor.submit(load_npz_file, npz_file)

        num_samples = binaryInputNCHW.shape[0]
        # Just discard stuff that doesn't divide evenly
        num_whole_steps = num_samples // (batch_size * world_size)

        #logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")
        for n in range(num_whole_steps):
            start = (n * world_size + rank) * batch_size
            end = start + batch_size

            batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
            batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
            batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
            batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
            batch_scoreDistrN = torch.from_numpy(scoreDistrN[start:end]).to(device)
            batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)


            if randomize_symmetries != 1:
                if randomize_symmetries==2:
                    symm = 5 * int(rand.integers(0, 2))  # 0:no sym   5:x-flip   others are not allowed
                elif randomize_symmetries==4:
                    symm = 5 * int(rand.integers(0, 2)) + 2 * int(rand.integers(0, 2))
                elif randomize_symmetries==8:
                    symm = int(rand.integers(0,8))
                else:
                    raise ValueError("read_npz_training_data(): randomize_symmetries should be 1 (no symmetry), 2 (x-axis symmetry), or 4 (x-axis or y-axis symmetry), or 8 (all symmetry)")
                batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm)
                batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len)
                batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm)

            value_sampling = (torch.rand(size=[end - start]) < value_sampling_proportion).float()
            if(value_sampling_proportion>0):
                value_sampling=value_sampling/value_sampling_proportion
            value_sampling=value_sampling.to(device)
            batch_globalTargetsNC[:,24] = value_sampling

            batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
            batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
            batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()
            
            batch = dict(
                binaryInputNCHW = batch_binaryInputNCHW,
                globalInputNC = batch_globalInputNC,
                policyTargetsNCMove = batch_policyTargetsNCMove,
                globalTargetsNC = batch_globalTargetsNC,
                scoreDistrN = batch_scoreDistrN,
                valueTargetsNCHW = batch_valueTargetsNCHW,
            )
            yield batch


def apply_symmetry_policy(tensor, symm, pos_len):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)

