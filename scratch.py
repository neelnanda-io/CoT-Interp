# %%
from neel.imports import *
import transformer_lens
from transformer_lens import (
    HookedTransformerConfig,
    HookedTransformer,
    FactoredMatrix,
    ActivationCache,
)
import sae_lens
from sae_lens import HookedSAETransformer
from neel_plotly import *

# %%
torch.set_grad_enabled(False)
# %%
model = HookedSAETransformer.from_pretrained("gemma-2-9b-it", dtype="bfloat16")
d_model = model.cfg.d_model
d_head = model.cfg.d_head
n_layers = model.cfg.n_layers
print(model.cfg)
# %%
import yaml

with open("/workspace/SAELens/sae_lens/pretrained_saes.yaml", "r") as file:
    pretrained_saes = yaml.safe_load(file)
print(pretrained_saes.keys())

RELEASE = "gemma-scope-9b-it-res-canonical"
saes = {}
for layer in [9, 20, 31]:
    for width in [16]:
        saes[(layer, width)] = sae_lens.SAE.from_pretrained(
            release=RELEASE,
            sae_id=f"layer_{layer}/width_{width}k/canonical",
            device="cuda",
        )[0].to(torch.float32)

# %%
START_TOKENS = torch.tensor([2, 106, 108], device="cuda")
MID_TOKENS = torch.tensor([107, 108, 106, 2516, 108], device="cuda")
def make_prompt(user_prompt, assistant_prompt=None):
    user_tokens = model.to_tokens(user_prompt).squeeze(0)
    if assistant_prompt is None:
        return torch.cat([START_TOKENS, user_tokens[1:], MID_TOKENS], dim=0)[None]
    else:
        return torch.cat([START_TOKENS, user_tokens[1:], MID_TOKENS, model.to_tokens(assistant_prompt).squeeze(0)[1:]], dim=0)[None]

prompt = "Calculate the area of a trapezoid with bases of 8 cm and 12 cm, and a height that is one-third the length of the longer base."
tokens = make_prompt(prompt)
step_by_step_tokens = make_prompt(prompt, "Let's think about this step by step.")
print(model.to_str_tokens(tokens))
print(model.to_str_tokens(step_by_step_tokens))
print()
print(model.generate(tokens, max_new_tokens=100, return_type="str"))
print()
print(model.generate(step_by_step_tokens, max_new_tokens=100, return_type="str"))
# %%
generated_tokens = model.generate(step_by_step_tokens, max_new_tokens=100, return_type="tensor").squeeze(0)
print(model.to_string(generated_tokens))

# %%
logits = model(generated_tokens)
nutils.show_df(nutils.create_vocab_df(logits[0, 55]).head(20))
nutils.show_df(nutils.create_vocab_df(logits[0, 57]).head(20))

# %%
area_prompt = """<start_of_turn>
Calculate the area of a trapezoid with bases of 8 cm and 12 cm, and a height that is one-third the length of the longer base.<end_of_turn>
<start_of_turn>model
Let's think about this step by step.

**1. Calculate the height:**

* The height is one-third the length of the longer base (12 cm).
* Height = (1/3) * 12 cm = 4 cm

**2. Consider the"""
perimeter_prompt = """<start_of_turn>
Calculate the perimeter of a trapezoid with bases of 8 cm and 12 cm, and a height that is one-third the length of the longer base.<end_of_turn>
<start_of_turn>model
Let's think about this step by step.

**1. Calculate the height:**

* The height is one-third the length of the longer base (12 cm).
* Height = (1/3) * 12 cm = 4 cm

**2. Consider the"""
area_tokens = model.to_tokens(area_prompt).squeeze(0)
perimeter_tokens = model.to_tokens(perimeter_prompt).squeeze(0)

x = model.generate(einops.repeat(area_tokens, "n -> 10 n"), max_new_tokens=10, return_type="tensor")
for i in x:
    print(model.to_string(i)[1+len(area_prompt):])

x = model.generate(einops.repeat(perimeter_tokens, "n -> 10 n"), max_new_tokens=10, return_type="tensor")
for i in x:
    print(model.to_string(i)[1+len(perimeter_prompt):])

# %%

area_logits = model(area_tokens)
nutils.show_df(nutils.create_vocab_df(area_logits[0, -1], make_probs=True).head(15))
perimeter_logits = model(perimeter_tokens)
nutils.show_df(nutils.create_vocab_df(perimeter_logits[0, -1], make_probs=True).head(15))


# %%
def get_cache_fwd_and_bwd(model, tokens, metric, layers):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    for layer in layers:
        model.add_hook(f"blocks.{layer}.hook_resid_post", forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    for layer in layers:
        model.add_hook(f"blocks.{layer}.hook_resid_post", backward_cache_hook, "bwd")
    torch.set_grad_enabled(True)
    value = metric(model(tokens.clone()))
    value.backward()
    torch.set_grad_enabled(False)
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )

# KEY_WORDS = ["Find", "Calculate", "Height"]
# KEY_TOKENS = [model.to_single_token(word) for word in KEY_WORDS]
# def log_prob_key_words(logits):
#     probs = F.softmax(logits.squeeze(0)[-1], dim=-1)
#     return probs[KEY_TOKENS].sum().log()

AREA_WORD = " formula"
PERIMETER_WORD = " other"
AREA_TOKEN = model.to_single_token(AREA_WORD)
PERIMETER_TOKEN = model.to_single_token(PERIMETER_WORD)

def patching_metric(logits):
    return logits[0, -1, AREA_TOKEN] - logits[0, -1, PERIMETER_TOKEN]


KEY_LAYERS = [9, 20, 31]
area_loss, area_fwd_cache, area_bwd_cache = get_cache_fwd_and_bwd(
    model, area_tokens, patching_metric, KEY_LAYERS
)
perimeter_loss, perimeter_fwd_cache, perimeter_bwd_cache = get_cache_fwd_and_bwd(
    model, perimeter_tokens, patching_metric, KEY_LAYERS
)
print(area_loss, perimeter_loss)
# %%
# Area section
area_resids = torch.stack([area_fwd_cache["resid_post", layer].squeeze(0) for layer in KEY_LAYERS])
area_grad_resids = torch.stack(
    [area_bwd_cache["resid_post", layer].squeeze(0) for layer in KEY_LAYERS]
)

width = 16
area_sae_acts_list = []
area_sae_attrs_list = []
for c, layer in enumerate(KEY_LAYERS):
    area_recons_resids, area_sae_cache = saes[(layer, width)].run_with_cache(area_resids[c].float())
    area_sae_acts = area_sae_cache[f"hook_sae_acts_post"]
    area_sae_attrs = (area_grad_resids[c].float() @ saes[(layer, width)].W_dec.T) * area_sae_acts
    area_sae_acts_list.append(area_sae_acts)
    area_sae_attrs_list.append(area_sae_attrs)
area_sae_acts = torch.stack(area_sae_acts_list)
area_sae_attrs = torch.stack(area_sae_attrs_list)


# Perimeter section
perimeter_resids = torch.stack([perimeter_fwd_cache["resid_post", layer].squeeze(0) for layer in KEY_LAYERS])
perimeter_grad_resids = torch.stack(
    [perimeter_bwd_cache["resid_post", layer].squeeze(0) for layer in KEY_LAYERS]
)

width = 16
perimeter_sae_acts_list = []
perimeter_sae_attrs_list = []
for c, layer in enumerate(KEY_LAYERS):
    perimeter_recons_resids, perimeter_sae_cache = saes[(layer, width)].run_with_cache(perimeter_resids[c].float())
    perimeter_sae_acts = perimeter_sae_cache[f"hook_sae_acts_post"]
    perimeter_sae_attrs = (perimeter_grad_resids[c].float() @ saes[(layer, width)].W_dec.T) * perimeter_sae_acts
    perimeter_sae_acts_list.append(perimeter_sae_acts)
    perimeter_sae_attrs_list.append(perimeter_sae_attrs)
perimeter_sae_acts = torch.stack(perimeter_sae_acts_list)
perimeter_sae_attrs = torch.stack(perimeter_sae_attrs_list)

# %%
width = 16
perimeter_sae_grads_list = []
area_sae_grads_list = []
for c, layer in enumerate(KEY_LAYERS):
    perimeter_sae_grads = (
        perimeter_grad_resids[c].float() @ saes[(layer, width)].W_dec.T
    )
    area_sae_grads = (
        area_grad_resids[c].float() @ saes[(layer, width)].W_dec.T
    )
    perimeter_sae_grads_list.append(perimeter_sae_grads)
    area_sae_grads_list.append(area_sae_grads)
perimeter_sae_grads = torch.stack(perimeter_sae_grads_list)
area_sae_grads = torch.stack(area_sae_grads_list)
perimeter_sae_grads.shape
# %%
p2a_sae_attrs = (area_sae_grads) * (area_sae_acts - perimeter_sae_acts)
a2p_sae_attrs = (perimeter_sae_grads) * (area_sae_acts - perimeter_sae_acts)
# %%
for sae_attrs, name in [(p2a_sae_attrs, "Perimeter to Area"), (a2p_sae_attrs, "Area to Perimeter")]:
    line(
        sae_attrs.sum(-1),
        x=nutils.process_tokens_index(area_tokens),
        color=[str(l) for l in KEY_LAYERS],
        title=name+" SAE Attribution Scores (sum)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )
    line(
        sae_attrs.max(-1).values,
        x=nutils.process_tokens_index(area_tokens),
        color=[str(l) for l in KEY_LAYERS],
        title=name+" SAE Attribution Scores (max)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )
    line(
        sae_attrs.min(-1).values,
        x=nutils.process_tokens_index(area_tokens),
        color=[str(l) for l in KEY_LAYERS],
        title=name+" SAE Attribution Scores (min)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )

    line(
        sae_attrs[:, 1:].sum(1),
        # x=nutils.process_tokens_index(generated_tokens[:56]),
        color=[str(l) for l in KEY_LAYERS],
        title=name+" SAE Attribution Scores (sum)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )
    line(
        sae_attrs[:, 1:].max(1).values,
        # x=nutils.process_tokens_index(generated_tokens[:56]),
        color=[str(l) for l in KEY_LAYERS],
        title=name+" SAE Attribution Scores (max)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )
    line(
        sae_attrs[:, 1:].min(1).values,
        # x=nutils.process_tokens_index(generated_tokens[:56]),
        color=[str(l) for l in KEY_LAYERS],
        title=name+" SAE Attribution Scores (min)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )
# %%
from IPython.display import IFrame


def get_dashboard_html(sae_release, sae_id, feature_idx):
    return f"https://neuronpedia.org/{sae_release}/{sae_id}/{feature_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"


feature_idx = 4773
layer = 20
width = 16

nutils.create_html(model.to_str_tokens(generated_tokens[:56]), sae_attrs[KEY_LAYERS.index(layer), :, feature_idx])

release, sae_id = saes[(layer, width)].cfg.neuronpedia_id.split("/")
html = get_dashboard_html(sae_release=release, sae_id=sae_id, feature_idx=feature_idx)
display(IFrame(html, width=1200, height=600))
# %%
for l in reversed(KEY_LAYERS):
    args = (-sae_attrs[l, 1:].sum(0)).argsort()[:3]
    print(l, )

# %%
rows = []
SAE_WIDTH = 16384
p2a_sae_attrs_np = utils.to_numpy(p2a_sae_attrs)
a2p_sae_attrs_np = utils.to_numpy(a2p_sae_attrs)
area_sae_attrs_np = utils.to_numpy(area_sae_attrs)
perimeter_sae_attrs_np = utils.to_numpy(perimeter_sae_attrs)
area_sae_acts_np = utils.to_numpy(area_sae_acts)
perimeter_sae_acts_np = utils.to_numpy(perimeter_sae_acts)
for li, l in enumerate(KEY_LAYERS):
    for pos in tqdm.trange(len(area_tokens)):
        str_token = model.to_string(area_tokens[pos])
        for latent in range(SAE_WIDTH):
            record = {
                "layer": l,
                "pos": pos,
                "latent": latent,
                "token": str_token,
                "p2a": p2a_sae_attrs_np[li, pos, latent],
                "a2p": a2p_sae_attrs_np[li, pos, latent],
                "area": area_sae_attrs_np[li, pos, latent],
                "per": perimeter_sae_attrs_np[li, pos, latent],
                "area_acts": area_sae_acts_np[li, pos, latent],
                "per_acts": perimeter_sae_acts_np[li, pos, latent],
            }
            rows.append(record)


df = pd.DataFrame(rows)
df.head()
# %%
trapez_df = df[df.token == " trapez"]
nutils.show_df(trapez_df.sort_values("a2p", ascending=False).head(10))
nutils.show_df(trapez_df.sort_values("area_acts", ascending=False).head(10))
nutils.show_df(trapez_df.sort_values("per_acts", ascending=False).head(10))
# %%
(trapez_df["per_acts"]>0).sum()
# %%
def get_attn_cache_fwd_and_bwd(model, tokens, metric, layers):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    for layer in layers:
        model.add_hook(f"blocks.{layer}.attn.hook_pattern", forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    for layer in layers:
        model.add_hook(f"blocks.{layer}.attn.hook_pattern", backward_cache_hook, "bwd")
    torch.set_grad_enabled(True)
    value = metric(model(tokens.clone()))
    value.backward()
    torch.set_grad_enabled(False)
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )
area_loss, area_fwd_cache_attn, area_bwd_cache_attn = get_attn_cache_fwd_and_bwd(model, area_tokens, patching_metric, list(range(n_layers)))
per_loss, per_fwd_cache_attn, per_bwd_cache_attn = get_attn_cache_fwd_and_bwd(model, perimeter_tokens, patching_metric, list(range(n_layers)))
# %%
area_attn = area_fwd_cache_attn.stack_activation("attn").squeeze()
per_attn = per_fwd_cache_attn.stack_activation("attn").squeeze()
area_attn_grads = area_bwd_cache_attn.stack_activation("attn").squeeze()
per_attn_grads = per_bwd_cache_attn.stack_activation("attn").squeeze()
area_attr = area_attn * area_attn_grads
per_attr = per_attn * per_attn_grads
# %%
LABELED_TOKENS = nutils.process_tokens_index(area_tokens)
imshow(area_attr.sum([0, 1]), x=LABELED_TOKENS, y=LABELED_TOKENS, title="Area SAE Attribution Scores")
imshow(per_attr.sum([0, 1]), x=LABELED_TOKENS, y=LABELED_TOKENS, title="Perimeter SAE Attribution Scores")
# %%
imshow(
    area_attr.std([2, 3]),
    # x=LABELED_TOKENS,
    # y=LABELED_TOKENS,
    title="Area SAE Attribution Scores",
)
imshow(
    per_attr.std([2, 3]),
    # x=LABELED_TOKENS,

    # y=LABELED_TOKENS,
    title="Perimeter SAE Attribution Scores",
)
# %%

