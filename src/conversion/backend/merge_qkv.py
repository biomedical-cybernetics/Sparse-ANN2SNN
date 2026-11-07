import re,torch
from copy import deepcopy
from typing import Dict,Any,Optional

def merge_qkv_in_state_dict(state_dict: Dict[str, torch.Tensor],
                            verbose: bool = True) -> Dict[str, torch.Tensor]:
    """
    在给定的 state_dict（扁平化键 -> Tensor）里：
      - 查找所有以 '.q_proj.weight' 为后缀的键
      - 对每个前缀（key_prefix），如果同时存在 key_prefix+'.k_proj.weight' 和 key_prefix+'.v_proj.weight'
        则将三者按顺序 [q;k;v] 在第0维拼接成 qkv.weight (shape (3*dim, dim))
      - 如果三者的 bias 都存在，则同样合并成 qkv.bias
      - 删除原来的 q_proj/k_proj/v_proj 的键，插入 qkv.* 键
    返回一个新的 state_dict（不会修改传入的 state_dict）。
    """
    sd = deepcopy(state_dict)  # 不直接修改原 dict，安全
    # 找出所有可能的前缀（匹配 .q_proj.weight 结尾）
    q_proj_pattern = re.compile(r'^(.*)\.q_proj\.weight$')
    prefixes = set()
    for k in state_dict.keys():
        m = q_proj_pattern.match(k)
        if m:
            prefixes.add(m.group(1))

    if verbose:
        print(f"Found {len(prefixes)} candidate prefixes with '.q_proj.weight'")

    converted = 0
    for prefix in sorted(prefixes):
        k_qw = prefix + '.q_proj.weight'
        k_kw = prefix + '.k_proj.weight'
        k_vw = prefix + '.v_proj.weight'

        # require all three weight keys present
        if not (k_qw in state_dict and k_kw in state_dict and k_vw in state_dict):
            if verbose:
                missing = [name for name in (k_qw, k_kw, k_vw) if name not in state_dict]
                print(f"Skipping {prefix}: missing weights {missing}")
            continue

        Wq = state_dict[k_qw]
        Wk = state_dict[k_kw]
        Wv = state_dict[k_vw]

        # sanity checks
        if Wq.ndim != 2 or Wk.ndim != 2 or Wv.ndim != 2:
            print(f"Skipping {prefix}: expected 2D weight matrices, got shapes {Wq.shape},{Wk.shape},{Wv.shape}")
            continue
        if not (Wq.shape[1] == Wk.shape[1] == Wv.shape[1]):
            print(f"Skipping {prefix}: in_features mismatch {Wq.shape},{Wk.shape},{Wv.shape}")
            continue
        if not (Wq.shape[0] == Wk.shape[0] == Wv.shape[0]):
            # This can happen if original q_proj dims differ; still allowed but we expect equal head dims
            # If shapes differ, you may want to still concatenate, but ensure it's intended.
            print(f"Warning {prefix}: out_features differ among q/k/v: {Wq.shape[0]},{Wk.shape[0]},{Wv.shape[0]}")

        # concat along 0 -> (3*dim_out, dim_in)
        qkv_w = torch.cat([Wq, Wk, Wv], dim=0)

        # biases
        k_qb = prefix + '.q_proj.bias'
        k_kb = prefix + '.k_proj.bias'
        k_vb = prefix + '.v_proj.bias'
        have_all_bias = (k_qb in state_dict) and (k_kb in state_dict) and (k_vb in state_dict)

        if have_all_bias:
            bq = state_dict[k_qb]
            bk = state_dict[k_kb]
            bv = state_dict[k_vb]
            if not (bq.ndim == bk.ndim == bv.ndim == 1):
                print(f"Skipping bias merge for {prefix}: unexpected bias shapes {bq.shape},{bk.shape},{bv.shape}")
                have_all_bias = False
            else:
                qkv_b = torch.cat([bq, bk, bv], dim=0)
        else:
            qkv_b = None

        # Insert new keys
        k_qkv_w = prefix + '.qkv.weight'
        sd[k_qkv_w] = qkv_w
        if qkv_b is not None:
            k_qkv_b = prefix + '.qkv.bias'
            sd[k_qkv_b] = qkv_b

        # Remove original keys (both weight and bias if present)
        for k in (k_qw, k_kw, k_vw, k_qb, k_kb, k_vb):
            if k in sd:
                del sd[k]

        converted += 1
        if verbose:
            if qkv_b is not None:
                print(f"Merged {prefix} -> inserted {k_qkv_w} and {k_qkv_b}")
            else:
                print(f"Merged {prefix} -> inserted {k_qkv_w} (no bias merged)")

    if verbose:
        print(f"Total merged prefixes: {converted}")
    return sd
