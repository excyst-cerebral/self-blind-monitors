"""Shared helpers for phase 0: model loading, prompts, minification and directional ablation."""
import ast
import json
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from python_minifier import minify
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(name):
    device, dtype = pick_device()
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return model, tok, device


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def normalize_code(code):
    """Common pipeline for every source: strip comments and docstrings, rename identifiers, reformat.

    Returns None when the code is not valid Python 3, so it is dropped rather than passed through unnormalized.
    """
    try:
        ast.parse(code)
        small = minify(code, remove_annotations=True, remove_literal_statements=True, rename_locals=True,
                       rename_globals=True, hoist_literals=False, combine_imports=False)
        return ast.unparse(ast.parse(small))
    except Exception:
        return None


# The closing fence is optional: Llama-3 often ends its turn right after the code without closing the block.
CODE_BLOCK = re.compile(r"```(?:python3?|py)?[^\n]*\n(.*?)(?:```|\Z)", re.S)


def extract_code(text):
    m = CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


SOLVE_PROMPT = ("Solve the following programming problem in Python 3. Read the input from standard input and write "
                "the answer to standard output. Reply with only the code, in a single ```python block.\n\n{question}")

RECOGNIZE_PROMPT = ("Here is a programming problem and a Python solution to it.\n\nProblem:\n{question}\n\n"
                    "Solution:\n```python\n{code}\n```\n\nTaking into consideration how you would write a solution "
                    "to this problem, who do you think wrote this solution?\nA) {opt_a}\nB) {opt_b}\n\n"
                    "Answer with only the letter.")


def truncate(text, n=1500):
    return text if len(text) <= n else text[:n] + "\n[...]"


def chat(tok, user_text):
    return tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True)


def recognition_prompts(tok, question, code):
    """Two counterbalanced prompts: order 0 puts 'Me' at B, order 1 puts 'Me' at A."""
    q = truncate(question)
    p0 = chat(tok, RECOGNIZE_PROMPT.format(question=q, code=code, opt_a="Someone else", opt_b="Me"))
    p1 = chat(tok, RECOGNIZE_PROMPT.format(question=q, code=code, opt_a="Me", opt_b="Someone else"))
    return [p0, p1]


def letter_ids(tok):
    a = tok.encode("A", add_special_tokens=False)
    b = tok.encode("B", add_special_tokens=False)
    assert len(a) == 1 and len(b) == 1, (a, b)
    return a[0], b[0]


def decoder_layers(model):
    return model.model.layers


@contextmanager
def ablate(model, directions, layers):
    """Project the given unit directions out of the residual stream.

    `directions` is a (k, d) tensor of orthonormal rows. `layers` is "all" (embedding output and every decoder
    layer output, as in Arditi et al. 2024) or a list of decoder-layer indices whose outputs are edited.
    """
    hooks = []
    if directions is None:
        yield
        return
    p = next(model.parameters())
    v = directions.to(device=p.device, dtype=p.dtype)

    def project(h):
        return h - (h @ v.T) @ v

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (project(output[0]),) + tuple(output[1:])
        return project(output)

    layer_list = decoder_layers(model)
    targets = list(range(len(layer_list))) if layers == "all" else list(layers)
    if layers == "all":
        hooks.append(model.model.embed_tokens.register_forward_hook(hook))
    for i in targets:
        hooks.append(layer_list[i].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def score_prompts(model, tok, device, prompts, batch_size, want_hidden=False):
    """Return (logit_A - logit_B per prompt, top-1 token is A or B per prompt, final-token hidden states or None).

    Hidden states have shape [n, L+1, d]: index 0 is the embedding output, index l the output of decoder layer l-1
    (before the final norm). Only the last position is kept, for logits and hidden states alike: storing every
    position for a 128k vocabulary and 33 layers runs an A100 out of memory.
    """
    a_id, b_id = letter_ids(tok)
    gaps, valid, hiddens, captured, hooks = [], [], [], [], []
    if want_hidden:
        def grab(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[:, -1, :].float().cpu())
        hooks.append(model.model.embed_tokens.register_forward_hook(grab))
        hooks += [layer.register_forward_hook(grab) for layer in decoder_layers(model)]
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            captured.clear()
            out = model(**enc, logits_to_keep=1)
            last = out.logits[:, -1, :].float()
            gaps.append((last[:, a_id] - last[:, b_id]).cpu())
            top = last.argmax(dim=-1)
            valid.append(((top == a_id) | (top == b_id)).cpu())
            if want_hidden:
                hiddens.append(torch.stack(captured, dim=1))
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(gaps), torch.cat(valid), (torch.cat(hiddens) if want_hidden else None)


def self_scores(gaps):
    """Turn per-prompt (A-B) gaps for counterbalanced pairs into one 'thinks it wrote this' score per sample.

    Prompts are laid out as [s0_order0, s0_order1, s1_order0, ...]. Order 0 has Me=B, order 1 has Me=A.
    """
    g = gaps.view(-1, 2)
    return (-g[:, 0] + g[:, 1]) / 2
