"""Batched MLP trainer: K independent copies of the 63-64-32-2 (here F-64-32-2) MLP
(Linear-BN-ReLU-Dropout(0.3) x2 -> Linear), Adam(lr=0.005), batch 32, cross-entropy,
trained in parallel with torch.bmm.  Every model has its own data, own initialisation and
own mini-batch order, so the K models are statistically independent (Adam is element-wise,
the total loss is the sum of the K per-model mean losses, hence gradients do not mix).

For every model we log, per epoch: train loss/acc, validation class-balanced CE, test CE/acc/
balanced-acc (test is NEVER used for model selection; it is logged for diagnostics only).
The model used for the "validated" result is the epoch with minimum validation balanced CE;
the "final" result is the model after the last epoch.
"""
import math, time
import numpy as np, torch

H1, H2 = 64, 32


def _init(K, F, g):
    def U(shape, fan_in):
        b = 1.0 / math.sqrt(fan_in)
        return (torch.rand(shape, generator=g) * 2 - 1) * b
    p = dict(
        W1=U((K, F, H1), F), b1=U((K, 1, H1), F),
        W2=U((K, H1, H2), H1), b2=U((K, 1, H2), H1),
        W3=U((K, H2, 2), H2), b3=U((K, 1, 2), H2),
        g1=torch.ones(K, 1, H1), be1=torch.zeros(K, 1, H1),
        g2=torch.ones(K, 1, H2), be2=torch.zeros(K, 1, H2))
    for v in p.values():
        v.requires_grad_(True)
    return p


def _forward(p, x, run, train, g=None, drop=0.3, mom=0.1, eps=1e-5):
    h = torch.baddbmm(p["b1"], x, p["W1"])
    h = _bn(h, p["g1"], p["be1"], run, "1", train, mom, eps)
    h = torch.relu(h)
    if train:
        h = h * (torch.rand(h.shape, generator=g) > drop).float() / (1 - drop)
    h = torch.baddbmm(p["b2"], h, p["W2"])
    h = _bn(h, p["g2"], p["be2"], run, "2", train, mom, eps)
    h = torch.relu(h)
    if train:
        h = h * (torch.rand(h.shape, generator=g) > drop).float() / (1 - drop)
    return torch.baddbmm(p["b3"], h, p["W3"])


def _bn(h, gam, bet, run, k, train, mom, eps):
    if train:
        m = h.mean(1, keepdim=True)
        v = h.var(1, unbiased=False, keepdim=True)
        n = h.shape[1]
        with torch.no_grad():
            run["m" + k].mul_(1 - mom).add_(mom * m)
            run["v" + k].mul_(1 - mom).add_(mom * v * n / (n - 1))
    else:
        m, v = run["m" + k], run["v" + k]
    return (h - m) / torch.sqrt(v + eps) * gam + bet


def _pad(arrs, dtype):
    K = len(arrs); n = max(len(a) for a in arrs)
    shp = arrs[0].shape[1:]
    out = np.zeros((K, n) + shp, dtype=dtype)
    mask = np.zeros((K, n), dtype=bool)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a; mask[i, :len(a)] = True
    return torch.from_numpy(out), torch.from_numpy(mask)


def _bal_ce(logits, y, mask):
    """class-balanced CE: mean over the two classes of the mean CE of that class"""
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), y.reshape(-1), reduction="none").reshape(y.shape)
    out = 0
    for c in (0, 1):
        m = mask & (y == c)
        out = out + (ce * m).sum(1) / m.sum(1).clamp(min=1)
    return out / 2, ce


def train_models(data, epochs=1000, seed=0, batch=32, lr=0.005, verbose=False):
    """data: list of dict(Xtr,ytr,Xva,yva,Xte,yte) (already standardised / balanced)."""
    K = len(data); F = data[0]["Xtr"].shape[1]
    g = torch.Generator().manual_seed(seed)
    Xtr, mtr = _pad([d["Xtr"] for d in data], np.float32)
    ytr, _ = _pad([d["ytr"] for d in data], np.int64)
    Xva, mva = _pad([d["Xva"] for d in data], np.float32)
    yva, _ = _pad([d["yva"] for d in data], np.int64)
    Xte, mte = _pad([d["Xte"] for d in data], np.float32)
    yte, _ = _pad([d["yte"] for d in data], np.int64)
    ntr = mtr.sum(1)
    S = int(math.ceil(ntr.max().item() / batch))
    p = _init(K, F, g)
    run = {"m1": torch.zeros(K, 1, H1), "v1": torch.ones(K, 1, H1),
           "m2": torch.zeros(K, 1, H2), "v2": torch.ones(K, 1, H2)}
    opt = torch.optim.Adam(list(p.values()), lr=lr)
    Nmax = Xtr.shape[1]
    ar = torch.arange(S * batch)
    rec = {k: np.zeros((K, epochs), dtype=np.float32) for k in
           ["tr_loss", "tr_acc", "va_bce", "te_ce", "te_acc", "te_bacc"]}
    best = torch.full((K,), float("inf")); best_ep = torch.zeros(K, dtype=torch.long)
    best_probs = torch.zeros(K, Xte.shape[1], 2)
    best_par = {k: v.detach().clone() for k, v in p.items()}
    best_run = {k: v.clone() for k, v in run.items()}
    t0 = time.time()
    for ep in range(epochs):
        r = torch.rand(K, Nmax, generator=g)
        r[~mtr] = 2.0
        perm = r.argsort(1)                                   # valid indices first
        pos = (ar[None, :] % ntr[:, None])                    # wrap-around for smaller sets
        order = perm.gather(1, pos).reshape(K, S, batch)
        tl = torch.zeros(K); ta = torch.zeros(K)
        for s in range(S):
            ix = order[:, s]
            xb = Xtr.gather(1, ix[:, :, None].expand(-1, -1, F))
            yb = ytr.gather(1, ix)
            lg = _forward(p, xb, run, True, g)
            l = torch.nn.functional.cross_entropy(lg.reshape(-1, 2), yb.reshape(-1), reduction="none").reshape(K, batch).mean(1)
            opt.zero_grad(set_to_none=True)
            l.sum().backward()
            opt.step()
            tl += l.detach(); ta += (lg.argmax(-1) == yb).float().mean(1)
        with torch.no_grad():
            lv = _forward(p, Xva, run, False)
            vb, _ = _bal_ce(lv, yva, mva)
            lt = _forward(p, Xte, run, False)
            tb, tce = _bal_ce(lt, yte, mte)
            pred = lt.argmax(-1)
            ok = (pred == yte) & mte
            acc = ok.sum(1) / mte.sum(1)
            rec0 = ((pred == 0) & (yte == 0) & mte).sum(1) / ((yte == 0) & mte).sum(1).clamp(min=1)
            rec1 = ((pred == 1) & (yte == 1) & mte).sum(1) / ((yte == 1) & mte).sum(1).clamp(min=1)
            rec["tr_loss"][:, ep] = (tl / S).numpy(); rec["tr_acc"][:, ep] = (ta / S).numpy()
            rec["va_bce"][:, ep] = vb.numpy()
            rec["te_ce"][:, ep] = ((tce * mte).sum(1) / mte.sum(1)).numpy()
            rec["te_acc"][:, ep] = acc.numpy(); rec["te_bacc"][:, ep] = ((rec0 + rec1) / 2).numpy()
            imp = vb < best
            if imp.any():
                best = torch.where(imp, vb, best); best_ep[imp] = ep
                best_probs[imp] = torch.softmax(lt, -1)[imp]
                for k in p:
                    best_par[k][imp] = p[k].detach()[imp]
                for k in run:
                    best_run[k][imp] = run[k][imp]
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  ep {ep+1}/{epochs}  {time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        final_probs = torch.softmax(_forward(p, Xte, run, False), -1)
    nte = [len(d["yte"]) for d in data]
    out = dict(rec=rec, best_ep=best_ep.numpy(),
               probs_sel=[best_probs[i, :nte[i]].numpy() for i in range(K)],
               probs_fin=[final_probs[i, :nte[i]].numpy() for i in range(K)],
               par_fin={k: v.detach().numpy() for k, v in p.items()},
               par_sel={k: v.numpy() for k, v in best_par.items()},
               run_sel={k: v.numpy() for k, v in best_run.items()},
               run_fin={k: v.numpy() for k, v in run.items()})
    return out


def numpy_forward(par, run, k, X):
    """inference-mode forward of model k (numpy) -> class-1 probability (used by SHAP)."""
    h = X @ par["W1"][k] + par["b1"][k]
    h = (h - run["m1"][k]) / np.sqrt(run["v1"][k] + 1e-5) * par["g1"][k] + par["be1"][k]
    h = np.maximum(h, 0)
    h = h @ par["W2"][k] + par["b2"][k]
    h = (h - run["m2"][k]) / np.sqrt(run["v2"][k] + 1e-5) * par["g2"][k] + par["be2"][k]
    h = np.maximum(h, 0)
    z = h @ par["W3"][k] + par["b3"][k]
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(-1, keepdims=True))[:, 1]
