import tempfile

import pandas as pd
import numpy as np
import cloudpickle
from tempfile import TemporaryDirectory
from pandas.core.internals.blocks import RemoteBlock

sample_csv_data = \
    """
    object_1,int64_1,float64_1
    a,1,1.2
    b,10,3.1
    c,99,20.0
    """

sample_data = {
        'int64_1': np.array([1, 2, 3], dtype=np.int64),
        'int64_2': np.array([10, 20, 30], dtype=np.int64),
        'int32_1': np.array([9, 8, 7], dtype=np.int32),
        'float64_1': np.array([.1, .5, .7], dtype=np.float64)
    }


def test_delegation():
    local = pd.DataFrame(sample_data)
    remote = pd.DataFrame(sample_data)
    remote_blocks = [RemoteBlock(blk, blk._mgr_locs, blk.shape, blk.ndim) for blk in remote._data.blocks]
    remote._data.blocks = tuple(remote_blocks)

    # Sanity Check
    for blk in remote._data.blocks:
        assert isinstance(blk, RemoteBlock)

    assert local.equals(remote)
    assert local.mean().equals(remote.mean())
    assert local['int32_1'].equals(remote['int32_1'])


def _test_lazy(local, remote):
    # Note: blknos and blkloc are lazily initialized. Unfortunately, this
    # causes unnecessary blocks to be loaded. A "hack" to avoid this is to
    # force the values to be set. (Another way to call print() but this is
    # likely cheaper). This also likely means that we shouldn't use
    # create_block_manager_from_blocks() and should serialize the original dataframe.
    #
    # Note: Remember to do this in the output stage of task compute elements
    # Update: Pushing up placement and ndim will address this
    #_ = remote._data.blknos
    #_ = remote._data.blklocs

    # Note: The debugger could break the load assertions since it'll trigger _load()

    def assert_load(df, expected):
        l = 0
        for b in df._data.blocks:
            assert isinstance(b, RemoteBlock)
            l += 1 if b.delegate is not None else 0

        assert l == expected

    with TemporaryDirectory() as td:
        remote_blocks_ser = []
        for blk in remote._data.blocks:
            serialized = cloudpickle.dumps(blk)
            with tempfile.NamedTemporaryFile(dir=td, delete=False) as f:
                f.write(serialized)
                remote_block_ser = {
                    "bp": blk._mgr_locs.orig_val,
                    "shape": blk.shape,
                    "ndim": blk.ndim,
                    "remote_meta": {"type": "lfs", "meta": {"path": f.name}}
                }
                dumped = cloudpickle.dumps(remote_block_ser)
                remote_blocks_ser.append(dumped)

        # We can
        remote._data.blocks = ()

        # Simulate deserializing DF w.o blocks
        with tempfile.NamedTemporaryFile(dir=td, delete=False) as f:
            f.write(cloudpickle.dumps(remote))
            f.flush()
            f.seek(0)
            remote = cloudpickle.loads(f.read())

        remote_blocks = []
        for blk_ser in remote_blocks_ser:
            loaded = cloudpickle.loads(blk_ser)
            remote_block = RemoteBlock(bp=loaded['bp'], shape=loaded['shape'], ndim=loaded['ndim'],
                                       remote_meta=loaded['remote_meta'])
            remote_blocks.append(remote_block)

        remote._data.blocks = tuple(remote_blocks)

        assert_load(remote, 0)
        #print(remote._data.blocks)

        assert local['float64_1'].equals(remote['float64_1'])
        assert local['float64_1'].mean() == remote['float64_1'].mean()

        assert_load(remote, 1)
        #print(remote._data.blocks)

        assert local.loc[:, ['int64_1', 'float64_1']].equals(remote.loc[:, ['int64_1', 'float64_1']])
        assert local.loc[:, ['int64_1', 'float64_1']].mean().equals(remote.loc[:, ['int64_1', 'float64_1']].mean())

        assert_load(remote, 2)
        #print(remote._data.blocks)

        # this seems to load all columns, but it's probably because of pandas implementation detail
        assert local[['int64_1', 'float64_1']].equals(remote[['int64_1', 'float64_1']])
        assert local[['int64_1', 'float64_1']].mean().equals(remote[['int64_1', 'float64_1']].mean())

        #print(remote._data.blocks)


def test_lazy():
    _test_lazy(pd.DataFrame(sample_data), pd.DataFrame(sample_data))


def test_lazy_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp:
        temp.write(bytes(sample_csv_data, encoding='utf-8'))
        temp.flush()
        _test_lazy(pd.read_csv(temp.name), pd.read_csv(temp.name))


test_delegation()
test_lazy()
test_lazy_csv()
