import os
import argparse
import numpy as np
import uproot
import awkward as ak
import vector
import fnmatch
import traceback

vector.register_awkward()

class WeaverPreprocessor:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()

    def process(self, chunk):
        def get(names):
            if isinstance(names, str): names = [names]
            for name in names:
                if name in chunk.fields: return chunk[name]
            raise KeyError(f"None of {names} found in file.")

        part_px = get(['part_px'])
        part_py = get(['part_py'])
        part_pz = get(['part_pz'])
        part_energy = get(['part_energy'])
        

        part_deta = get(['part_deta'])
        part_dphi = get(['part_dphi'])

        jet_pt = get(['jet_pt'])
        jet_energy = get(['jet_energy'])
        
        part_pt = np.hypot(part_px, part_py)
        part_pt_log = np.log(part_pt)
        part_e_log = np.log(part_energy)
        part_logptrel = np.log(part_pt / jet_pt)
        part_logerel = np.log(part_energy / jet_energy)
        part_deltaR = np.hypot(part_deta, part_dphi)
        
        part_zeros = ak.zeros_like(part_pt)

        def norm(arr, sub=0, mul=1, min_v=-5, max_v=5):
            x = (arr - sub) * mul
            x = ak.where(x < min_v, min_v, x)
            x = ak.where(x > max_v, max_v, x)
            return x

        if self.dataset_name == 'jetclass':
            pf_mask_raw = ak.ones_like(part_energy)
            
            part_d0 = np.tanh(get(['part_d0val']))
            part_dz = np.tanh(get(['part_dzval']))
            part_d0err = get(['part_d0err'])
            part_dzerr = get(['part_dzerr'])
            
            f_list = [
                norm(part_pt_log, 1.7, 0.7),
                norm(part_e_log, 2.0, 0.7),
                norm(part_logptrel, -4.7, 0.7),
                norm(part_logerel, -4.7, 0.7),
                norm(part_deltaR, 0.2, 4.0),
                get('part_charge'),
                get('part_isChargedHadron'),
                get('part_isNeutralHadron'),
                get('part_isPhoton'),
                get('part_isElectron'),
                get('part_isMuon'),
                part_d0, 
                norm(part_d0err, 0, 1, 0, 1),
                part_dz, 
                norm(part_dzerr, 0, 1, 0, 1),
                part_deta,
                part_dphi,
            ]
            
            l_cols = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 
                      'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
            labels = np.stack([ak.to_numpy(get(n)) for n in l_cols], axis=1)

        elif 'qg' in self.dataset_name or 'quark' in self.dataset_name:
            pf_mask_raw = ak.ones_like(part_deta)
            
            f_list = [
                norm(part_pt_log, 1.7, 0.7),
                norm(part_e_log, 2.0, 0.7),
                norm(part_logptrel, -4.7, 0.7),
                norm(part_logerel, -4.7, 0.7),
                norm(part_deltaR, 0.2, 4.0),
                get('part_charge'),
                get('part_isChargedHadron'),
                get('part_isNeutralHadron'),
                get('part_isPhoton'),
                get('part_isElectron'),
                get('part_isMuon'),
                part_zeros,
                part_zeros,
                part_zeros,
                part_zeros,
                part_deta,
                part_dphi,
            ]
            
            # Labels: [label, 1-label] (Quark, Gluon)
            lbl = get(['label'])
            lbl = ak.to_numpy(lbl)
            labels = np.stack([lbl, 1-lbl], axis=1)

        elif 'top' in self.dataset_name:
            pf_mask_raw = ak.ones_like(part_deta)
            
            f_list = [
                norm(part_pt_log, 1.7, 0.7),
                norm(part_e_log, 2.0, 0.7),
                norm(part_logptrel, -4.7, 0.7),
                norm(part_logerel, -4.7, 0.7),
                norm(part_deltaR, 0.2, 4.0),
                part_zeros, part_zeros, part_zeros, part_zeros, part_zeros,
                part_zeros, part_zeros, part_zeros, part_zeros, part_zeros,
                part_deta,
                part_dphi,
            ]
            
            # Labels: [label, 1-label] (Top, QCD)
            lbl = get(['label'])
            lbl = ak.to_numpy(lbl)
            labels = np.stack([lbl, 1-lbl], axis=1)

        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        def stack_pad(arr_list):
            padded = [ak.fill_none(ak.pad_none(x, 128, clip=True), 0) for x in arr_list]
            return np.stack([ak.to_numpy(p) for p in padded], axis=1).astype(np.float32)

        pf_features = stack_pad(f_list)
        pf_vectors = stack_pad([part_px, part_py, part_pz, part_energy])
        pf_mask = stack_pad([pf_mask_raw])

        return pf_features, pf_vectors, pf_mask, labels.astype(np.float32)

def process_split(data_dir, dataset_name, split, limit):
    print(f"\nProcessing {split} set for {dataset_name} (Limit: {limit})...")
    
    search_dir = os.path.join(data_dir, split)
    if not os.path.exists(search_dir): search_dir = data_dir 
        
    files = [os.path.join(search_dir, f) for f in os.listdir(search_dir) if f.endswith(('.root', '.parquet'))]
    if not files: raise FileNotFoundError(f"No files found in {search_dir}")
    
    samples_per_file = max(1, limit // len(files))
    
    processor = WeaverPreprocessor(dataset_name)
    all_feats, all_vecs, all_masks, all_labels = [], [], [], [] 
    
    branches = [
        'part_px', 'part_py', 'part_pz', 'part_energy', 
        'part_deta', 'part_dphi',
        'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy',
        'part_d0val', 'part_dzval', 'part_d0err', 'part_dzerr',
        'part_charge', 'part_isChargedHadron', 'part_isNeutralHadron',
        'part_isPhoton', 'part_isElectron', 'part_isMuon',
        'label*'
    ]

    def get_explicit_branches(file_path, patterns):
        if file_path.endswith('.root'):
            with uproot.open(file_path) as f:
                tree = f[f.keys()[0]]
                all_keys = tree.keys()
        elif file_path.endswith('.parquet'):
            ds = ak.from_parquet(file_path)
            all_keys = ds.fields
        else:
            return patterns

        explicit_branches = []
        for pattern in patterns:
            if '*' in pattern:
                matched = fnmatch.filter(all_keys, pattern)
                explicit_branches.extend(matched)
            else:
                if pattern in all_keys:
                    explicit_branches.append(pattern)
                elif pattern in ['PX_0', 'PY_0', 'PZ_0', 'E_0']:
                    pass 
                else:
                    pass

        return sorted(list(set(explicit_branches)))

    for fpath in files:
        try:
            real_branches = get_explicit_branches(fpath, branches)

            if fpath.endswith('.root'):
                with uproot.open(fpath) as f:
                    tree = f[f.keys()[0]]
                    for chunk in tree.iterate(real_branches, step_size=samples_per_file, library='ak'):
                        f, v, m, l = processor.process(chunk)
                        all_feats.append(f); all_vecs.append(v); all_masks.append(m); all_labels.append(l)
                        break
            elif fpath.endswith('.parquet'):
                ds = ak.from_parquet(fpath, columns=real_branches)
                chunk = ds[:samples_per_file] if len(ds) > samples_per_file else ds
                chunk = ak.to_packed(chunk)
                f, v, m, l = processor.process(chunk)
                all_feats.append(f); all_vecs.append(v); all_masks.append(m); all_labels.append(l)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            traceback.print_exc()

    if not all_labels: raise RuntimeError("No data loaded.")
    
    # Save
    save_dir = os.path.join(data_dir, f"{split}_processed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "pf_features.npy"), np.concatenate(all_feats))
    np.save(os.path.join(save_dir, "pf_vectors.npy"), np.concatenate(all_vecs))
    np.save(os.path.join(save_dir, "pf_mask.npy"), np.concatenate(all_masks))
    np.save(os.path.join(save_dir, "labels.npy"), np.concatenate(all_labels))
    print(f"Saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_limit', type=int, default=50000)
    parser.add_argument('--test_limit', type=int, default=50000)
    args = parser.parse_args()
    
    process_split(args.data_dir, args.dataset, 'train', args.train_limit)
    process_split(args.data_dir, args.dataset, 'test', args.test_limit)