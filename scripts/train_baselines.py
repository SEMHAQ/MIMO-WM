import torch, torch.nn as nn, numpy as np, os, json, glob

def load_eps(d, split, mx=None):
    dd = os.path.join(d, split)
    fs = sorted(glob.glob(os.path.join(dd, "episode_*.npz")))
    if mx: fs = fs[:mx]
    return [(np.load(f)['states'], np.load(f)['actions']) for f in fs]

def make_seqs(eps, T=32):
    Xs, Ya = [], []
    for states, actions in eps:
        for t in range(len(states) - T - 1):
            Xs.append(np.concatenate([states[t:t+T], actions[t:t+T]], axis=-1))
            Ya.append(states[t+1:t+T+1])
    return np.array(Xs), np.array(Ya)

class MLP_WM(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd+ad, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, sd))
    def forward(self, s, a):
        return self.net(torch.cat([s,a],-1))

class Linear_WM(nn.Module):
    def __init__(self, sd, ad):
        super().__init__()
        self.net = nn.Linear(sd+ad, sd)
    def forward(self, s, a):
        return self.net(torch.cat([s,a],-1))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
results = {}

for dataset in ['humanoid', 'ant']:
    dpath = f'data/{dataset}'
    train_eps = load_eps(dpath, 'train')
    val_eps = load_eps(dpath, 'val')
    sd = train_eps[0][0].shape[-1]
    ad = train_eps[0][1].shape[-1]
    
    Xs_tr, Ya_tr = make_seqs(train_eps, 32)
    Xs_val, Ya_val = make_seqs(val_eps, 32)
    print(f"{dataset}: train={len(Xs_tr)}, val={len(Xs_val)}, sd={sd}, ad={ad}")
    
    for mname, MC in [('MLP-WM', MLP_WM), ('Linear-WM', Linear_WM)]:
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = MC(sd, ad).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=5e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100)
            
            best_mse = 1e9
            for ep in range(100):
                model.train()
                idx = torch.randperm(len(Xs_tr))
                for i in range(0, len(Xs_tr), 64):
                    bi = idx[i:i+64]
                    pred = model(torch.FloatTensor(Xs_tr[bi,:,:sd]).to(device), 
                                torch.FloatTensor(Xs_tr[bi,:,sd:]).to(device))
                    target = torch.FloatTensor(Ya_tr[bi]).to(device)
                    loss = nn.MSELoss()(pred, target)
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                sched.step()
                
                if (ep+1) % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        pred = model(torch.FloatTensor(Xs_val[:2000,:,:sd]).to(device),
                                    torch.FloatTensor(Xs_val[:2000,:,sd:]).to(device))
                        target = torch.FloatTensor(Ya_val[:2000]).to(device)
                        mse = nn.MSELoss()(pred, target).item()
                        if mse < best_mse: best_mse = mse
            
            # Final eval
            model.eval()
            with torch.no_grad():
                pred = model(torch.FloatTensor(Xs_val[:,:,:sd]).to(device),
                            torch.FloatTensor(Xs_val[:,:,sd:]).to(device))
                target = torch.FloatTensor(Ya_val).to(device)
                mse = nn.MSELoss()(pred, target).item()
                ss_res = ((pred - target)**2).sum()
                ss_tot = ((target - target.mean())**2).sum()
                r2 = (1 - ss_res/ss_tot).item()
            
            key = f"{mname}_{dataset}"
            if key not in results: results[key] = {}
            results[key][f"seed{seed}"] = {"mse": round(mse, 6), "r2": round(r2, 6)}
            print(f"  {mname} {dataset} seed={seed}: MSE={mse:.6f}, R2={r2:.6f}")

with open('experiments/baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Done!")
