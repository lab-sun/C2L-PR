import numpy as np
import torch

import eval_f1
from net import RINet_attention


def _write_case(tmp_path):
    rng = np.random.RandomState(0)
    m = 4
    img_path = tmp_path / 'img.bin'
    img_path.write_bytes(rng.randn(m, 12, 360).astype(np.float32).tobytes())
    velo_paths = []
    for i in range(8):
        path = tmp_path / ('velo_%d.bin' % i)
        path.write_bytes(rng.randn(m, 12, 360).astype(np.float32).tobytes())
        velo_paths.append(str(path))
    pairs_path = tmp_path / 'pairs.txt'
    pairs = np.array([[0, 1, 1],
                      [1, 2, 0],
                      [2, 3, 1],
                      [3, 0, 0]], dtype=np.int32)
    np.savetxt(pairs_path, pairs, fmt='%d')
    torch.manual_seed(0)
    net = RINet_attention()
    ckpt = tmp_path / 'model.ckpt'
    torch.save({'state_dict': net.state_dict()}, ckpt)
    return str(ckpt), str(pairs_path), str(img_path), velo_paths


def _run(tmp_path, use_l2_dis):
    eval_f1.device = torch.device('cpu')
    ckpt, pairs, img, velos = _write_case(tmp_path)
    eval_f1.fast_eval(
        seq='00',
        model_file=ckpt,
        pair_file=pairs,
        use_l2_dis=use_l2_dis,
        img_desc_file=img,
        velo_desc_file_0=velos[0],
        velo_desc_file_1=velos[1],
        velo_desc_file_2=velos[2],
        velo_desc_file_3=velos[3],
        velo_desc_file_4=velos[4],
        velo_desc_file_5=velos[5],
        velo_desc_file_6=velos[6],
        velo_desc_file_7=velos[7],
    )


def test_fast_eval(tmp_path, capsys):
    _run(tmp_path, use_l2_dis=False)
    assert 'F1_split:' in capsys.readouterr().out


def test_fast_eval_l2(tmp_path, capsys):
    _run(tmp_path, use_l2_dis=True)
    assert 'F1_split:' in capsys.readouterr().out


def test_nan_view_does_not_drop_finite_l2_match():
    """One NaN orientation bin must not erase a finite match on another bin."""
    desc1 = np.array([[0, 0, 0, 0]], dtype=np.float32)
    finite = np.array([[1, 0, 0, 0]], dtype=np.float32)
    missing = np.full((1, 4), np.nan, dtype=np.float32)
    out = eval_f1.max_inverse_squared_l2(desc1, [missing, finite])
    assert out.shape == (1,)
    assert out[0] == 1.0
