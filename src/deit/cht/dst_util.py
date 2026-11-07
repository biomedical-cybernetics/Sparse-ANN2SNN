import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import sys

'''
def get_weighted_layers_Transformer(model, W_list, qk_chain_list, chain_list, args):
    def add_W(module):
        W_list.append(module.weight)
        module.LAYER_INDEX = len(W_list) - 1

    items = model._modules.items()

    for layer_name, p in items:
        if layer_name == 'generator':
            continue
        else:
            assert not isinstance(p, torch.nn.Linear)
            if layer_name in ["self_attn", "context_attn"]:
                add_W(p.linear_keys)
                add_W(p.linear_values)
                add_W(p.linear_query)
                add_W(p.final_linear)
                qk_chain_list.append([p.linear_keys.LAYER_INDEX, p.linear_query.LAYER_INDEX])
                chain_list.append([p.linear_values.LAYER_INDEX, p.final_linear.LAYER_INDEX])
            elif layer_name == "feed_forward":
                add_W(p.w_1)
                add_W(p.w_2)
                chain_list.append([p.w_1.LAYER_INDEX, p.w_2.LAYER_INDEX])
            else:
                get_weighted_layers_Transformer(p, W_list=W_list, qk_chain_list=qk_chain_list, chain_list=chain_list, args=args)
'''

def collect_vit_weight_matrices(model: nn.Module,
                                W_list: list,
                                qk_chain_list: list,
                                chain_list: list,
                                args
                                ) -> None:
    # 如果模型有 cls_token（ViT 常见），记住它的 id 以便跳过
    cls_token_obj = None
    if hasattr(model, 'cls_token'):
        cls_token_obj = getattr(model, 'cls_token')
        # 只有当它确实是 nn.Parameter 时我们跳过
        if not isinstance(cls_token_obj, nn.Parameter):
            cls_token_obj = None

    dist_token_obj = None
    if hasattr(model, 'dist_token'):
        dist_token_obj = getattr(model, 'dist_token')
        # 只有当它确实是 nn.Parameter 时我们跳过
        if not isinstance(dist_token_obj, nn.Parameter):
            dist_token_obj = None

    # 识别要跳过的 final head module（可能名称不同）
    # 把可能的 final head 对象记录下来（如果存在）
    final_head_modules = set()
    for name in ('head', 'head_dist'):
        if hasattr(model, name):
            print(f'Skip {name} to sparsify')
            attr = getattr(model, name)
            # 只跳过线性/模块类型的 head（例如 nn.Linear 或 nn.Module）
            if isinstance(attr, (nn.Module, nn.Linear)):
                final_head_modules.add(id(attr))

    def add_W(mod):
        if hasattr(mod, 'weight'):
            W_list.append(mod.weight)
        else:
            assert isinstance(mod, torch.Tensor)
            W_list.append(mod)

        return len(W_list)-1

    # 遍历所有子模块，按常见 ViT 结构进行匹配处理
    for name, module in model.named_modules():
        if id(module) in final_head_modules:
            continue
        if cls_token_obj is not None and id(module) == id(cls_token_obj):
                return
        if dist_token_obj is not None and id(module) == id(dist_token_obj):
                return

        # ATTENTION (timm ViT): has attributes qkv (Linear) and proj (Linear)
        if hasattr(module, 'q_proj') and hasattr(module, 'proj'):
            assert hasattr(module, 'k_proj') and hasattr(module, 'v_proj')
            proj = getattr(module, 'proj', None)
            q_proj = getattr(module, 'q_proj', None)
            k_proj = getattr(module, 'k_proj', None)
            v_proj = getattr(module, 'v_proj', None)
            idx_Wk = add_W(k_proj)
            idx_Wv = add_W(v_proj)
            idx_Wq = add_W(q_proj)
            idx_proj = add_W(proj)
            qk_chain_list.append([idx_Wk, idx_Wq])
            chain_list.append([idx_Wv, idx_proj])
            continue

        # MLP / FFN (timm ViT): typically has fc1 and fc2 (or named 'fc1','fc2' inside Mlp)
        if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
            fc1 = getattr(module, 'fc1', None)
            fc2 = getattr(module, 'fc2', None)
            idx_fc1 = add_W(fc1)
            idx_fc2 = add_W(fc2)
            chain_list.append([idx_fc1, idx_fc2])
            continue

def get_W(model, args):
    W_list = []
    qk_chain_list = []
    chain_list = []

    collect_vit_weight_matrices(model, W_list=W_list, qk_chain_list=qk_chain_list, chain_list=chain_list, args=args)


    return W_list, chain_list, qk_chain_list

def split_attention_qkv(attn_module: nn.Module, remove_qkv: bool = True) -> bool:
    """
    将 timm.Attention 的单个 Linear qkv 拆成 q_proj/k_proj/v_proj（三个 nn.Linear），
    并替换 forward。返回 True 表示转换成功。
    - attn_module: instance of timm Attention
    - remove_qkv: 如果 True，则把原先的 attn_module.qkv 设为 None（避免混淆）
    """
    # 基本检查
    if not hasattr(attn_module, 'qkv') or not isinstance(getattr(attn_module, 'qkv'), nn.Linear):
        return False

    qkv = attn_module.qkv
    device = qkv.weight.device
    dtype = qkv.weight.dtype

    # qkv weight 形状检查
    W = qkv.weight.data  # Tensor (3*dim, dim)
    if W.ndim != 2:
        raise RuntimeError(f"unexpected qkv weight dim: {W.shape}")
    out_dim, dim = W.shape
    if out_dim % 3 != 0 or out_dim != 3 * dim:
        # 有些实现会把 dim != out_dim/3，这里强检查以防万一
        # 但一般 timm 是 (3*dim, dim)
        raise RuntimeError(f"unexpected qkv shape: {W.shape}, expecting (3*dim, dim)")

    # 切片并 clone 权重/偏置（一次性操作，安全）
    d = dim
    Wq = W[:d, :].clone()
    Wk = W[d:2*d, :].clone()
    Wv = W[2*d:3*d, :].clone()

    b = qkv.bias.data if qkv.bias is not None else None
    if b is not None:
        bq = b[:d].clone()
        bk = b[d:2*d].clone()
        bv = b[2*d:3*d].clone()
    else:
        bq = bk = bv = None

    # 创建三个 Linear，保持 device/dtype
    q_proj = nn.Linear(dim, dim, bias=(bq is not None)).to(device=device, dtype=dtype)
    k_proj = nn.Linear(dim, dim, bias=(bk is not None)).to(device=device, dtype=dtype)
    v_proj = nn.Linear(dim, dim, bias=(bv is not None)).to(device=device, dtype=dtype)

    # 复制 weight / bias
    q_proj.weight.data.copy_(Wq)
    k_proj.weight.data.copy_(Wk)
    v_proj.weight.data.copy_(Wv)
    if bq is not None:
        q_proj.bias.data.copy_(bq)
        k_proj.bias.data.copy_(bk)
        v_proj.bias.data.copy_(bv)

    # attach new modules
    attn_module.q_proj = q_proj
    attn_module.k_proj = k_proj
    attn_module.v_proj = v_proj

    # optionally remove old qkv to avoid confusion
    if remove_qkv:
        attn_module.qkv = None

    # Prepare maybe_add_mask fallback:
    # try to get maybe_add_mask from the same module namespace where Attention is defined
    maybe_add_mask_fn = None
    try:
        mod = sys.modules.get(attn_module.__module__)
        if mod is not None and hasattr(mod, 'maybe_add_mask'):
            maybe_add_mask_fn = getattr(mod, 'maybe_add_mask')
    except Exception:
        maybe_add_mask_fn = None

    if maybe_add_mask_fn is None:
        # fallback simple implementation (works for most masks that are additive/broadcastable)
        def maybe_add_mask_fn(attn, mask):
            return attn if mask is None else attn + mask

    # Define new forward that mirrors the original Attention.forward behavior
    def new_forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape

        # produce q, k, v with shape (B, num_heads, N, head_dim)
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # q,k normalization layers expect last dim = head_dim
        q = self.q_norm(q)
        k = self.k_norm(k)

        # fused path (use PyTorch scaled_dot_product_attention if available)
        if getattr(self, 'fused_attn', False):
            # attn_drop.p exists on the Dropout module
            dropout_p = self.attn_drop.p if getattr(self.attn_drop, 'p', None) is not None and self.training else 0.0
            x_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
            )
        else:
            q_scaled = q * self.scale
            attn = q_scaled @ k.transpose(-2, -1)  # shape (B, num_heads, N, N)
            attn = maybe_add_mask_fn(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_out = attn @ v

        # combine heads
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.norm(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    # Bind new_forward to the instance
    attn_module.forward = types.MethodType(new_forward, attn_module)

    return True

def recur_split_atten(model):
    for name, m in model.named_modules():
        if hasattr(m, 'qkv') and isinstance(getattr(m, 'qkv'), nn.Linear):
            if split_attention_qkv(m):
                print("Converted attention:", name)
