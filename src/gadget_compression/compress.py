import h5py
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import hdf5plugin
import numpy as np
import typer

app = typer.Typer()

prefix = "snap_"
nworkers = 8
precision_default = 8  # for zfp
pointwise_relative_default = 0.05  # for sz

def get_compression(dset):
    key = dset.name
    if "int" in str(dset.dtype):
        compression = hdf5plugin.Blosc2(cname="zstd", clevel=4, filters=1)
        return compression
    if key.endswith("Coordinates") or key.endswith("Pos"):  # dont care about CoM...: "or key.endswith("CenterOfMass"):"
        # dont want any lossy compression here
        compression = hdf5plugin.Blosc2(cname="zstd", clevel=4, filters=1)
        return compression
    #compression = hdf5plugin.Zfp(precision=precision_default)
    compression=hdf5plugin.SZ(pointwise_relative=pointwise_relative_default)
    return compression


def walk_h5(grp, abspath=None, func_grp=None, func_dset=None):
    if abspath is None:
        abspath = grp.name
    for key in grp.keys():
        abspath_new = abspath + "/" + key
        if isinstance(grp[key], h5py.Group):
            if func_grp is not None:
                func_grp(grp[key], abspath=abspath_new)
            walk_h5(grp[key], abspath=abspath_new, func_grp=func_grp, func_dset=func_dset)
        else:
            if isinstance(grp[key], h5py.Dataset):
                if func_dset is not None:
                    func_dset(grp[key], abspath=abspath_new)

def compress_file(file, inputpath, outputpath):
    # walk h5 file
    dsets = []
    attrs = []

    # make sure outputpath exists
    if not os.path.exists(outputpath):
        try:
            os.makedirs(outputpath)
        except FileExistsError:
            pass
    # create new file
    hf_out = h5py.File(os.path.join(outputpath, file), "w")

    # copy attributes over from given grp/dset to new file
    def copy_attrs(src, dst, dstpath=None):
        for key, value in src.attrs.items():
            dst_target = dst
            if dstpath is not None:
                dst_target = dst[dstpath]
            dst_target.attrs[key] = value

    copy_attrs_func = partial(copy_attrs, dst=hf_out)

    def func_dset(dset, abspath=None):
        compression = get_compression(dset)
        #print(dset.name, compression)
        hf_out.create_dataset(abspath, data=dset, compression=compression)
        #copy_attrs_func(dset, dstpath=dset.name)

    def func_grp(grp, abspath=None):
        #print(grp.name)
        hf_out.create_group(grp.name)
        copy_attrs_func(grp, dstpath=grp.name)

    with h5py.File(os.path.join(inputpath, file), "r") as f:
        walk_h5(f, func_dset=func_dset, func_grp=func_grp)

# compare compressed vs uncompressed
def compare_fields(file, field, inputpath, outputpath, op=np.max):
    with h5py.File(os.path.join(inputpath, file), "r") as f:
        data_in = f[field][:]
    with h5py.File(os.path.join(outputpath, file), "r") as f:
        data_out = f[field][:]
    res = op(np.abs(data_in - data_out) / np.abs(data_in))
    return res

@app.command()
def compress_snapshot(path_in: str, path_out: str):
    # check if path_in is a folder
    if not os.path.isdir(path_in):
        assert os.path.isfile(path_in)
        # take base file in folder
        filename = os.path.basename(path_in)
        path_in = os.path.dirname(path_in)
        files = [filename]
    else:
        files = [f for f in os.listdir(path_in) if f.startswith(prefix)]

    nbytes_in_list = []
    nbytes_out_list = []

    mapfunc = partial(compress_file, inputpath=path_in, outputpath=path_out)

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        tasks = executor.map(mapfunc, files)

    for task in tasks:
        pass  # just to check for exceptions

    for file in files:
        nbytes_in_list.append(os.path.getsize(os.path.join(path_in, file)))
        nbytes_out_list.append(os.path.getsize(os.path.join(path_out, file)))

    nbytes_in = sum(nbytes_in_list)
    nbytes_out = sum(nbytes_out_list)

    print(f"Compression ratio: {nbytes_in/nbytes_out:.2f}")



if __name__ == "__main__":
    #inputpath = "/newdata/data/public/testdata-astrodask/tng50subbox0_snap1216"
    #outputpath = "./compressed"
    #compress_snapshot(inputpath, outputpath)

    #files = [f for f in os.listdir(inputpath) if f.startswith(prefix)]
    #for file in files:
    #    relmax = compare_fields(file, "/PartType0/Density", inputpath, outputpath, op=np.max)
    #    mean = compare_fields(file, "/PartType0/Density", inputpath, outputpath, op=np.mean)
    #    std = compare_fields(file, "/PartType0/Density", inputpath, outputpath, op=np.std)
    #    #print(f"relative Max for file {file}: {relmax:e}")
    #    #print(f"relative Mean for file {file}: {mean:e}")
    #    print(f"relative Std for file {file}: {std:e}")
    app()

