import torch

from net import RINet_attention


def test_gen_feature_shape():
    torch.manual_seed(0)
    net = RINet_attention().eval()
    n = 4
    x = torch.randn(n, 12, 360)
    with torch.no_grad():
        feat = net.gen_feature(x)
    assert feat.shape == (n, 288)


def test_forward_shapes_and_eval_scores():
    torch.manual_seed(0)
    net = RINet_attention().eval()
    n = 4
    query = torch.randn(n, 12, 360)
    descs = [torch.randn(n, 12, 360) for _ in range(8)]
    with torch.no_grad():
        out, diff, out_cat = net(query, *descs)
    assert out.shape == (n,)
    assert diff.shape == (n,)
    assert out_cat.shape == (8, n)
    assert torch.all(out >= 0) and torch.all(out <= 1)
    assert torch.all(out_cat >= 0) and torch.all(out_cat <= 1)


def test_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = RINet_attention().eval()
    n = 3
    query = torch.randn(n, 12, 360)
    descs = [torch.randn(n, 12, 360) for _ in range(8)]
    with torch.no_grad():
        ref = net(query, *descs)
    ckpt = tmp_path / 'model.ckpt'
    torch.save({'state_dict': net.state_dict()}, ckpt)
    loaded = RINet_attention().eval()
    loaded.load(str(ckpt))
    with torch.no_grad():
        out = loaded(query, *descs)
    for got, expected in zip(out, ref):
        assert torch.equal(got, expected)
