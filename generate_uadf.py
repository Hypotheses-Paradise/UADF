import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import lazy_load, llama_model_lookup, quantization

# Modify ./generate/*.py to fit the requirement of UADF decoding 

@torch.no_grad()
def generate(
    model: LLaMA,
    idx: torch.Tensor,
    asr_emb,                     # acoustic representations from the ASR encoder
    asr_inference,               # a pre-trained ASR model's decoder aligning wiht LLM's tokenizer
    max_new_tokens: int,
    *,
    max_seq_length: Optional[int] = None,
    temperature_asr=1.0,        # \tau_1 for asr calibration, obtained by binary search
    temperature_llm=1.0,        # \tau_2 for llm calibration, obtained by binary search
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    # generate max_new_tokens tokens
    decoder_input = torch.tensor([[1]]).long().to(device)
    sigmoid = torch.nn.Sigmoid()

    for _ in range(max_new_tokens):
        # asr inference, calibration, and softmax
        asr_logits, _ = asr_inference(decoder_input, asr_emb)  / temperature_asr
        asr_prob = torch.nn.functional.softmax(asr_logits, dim=-1)

        # llm inference, calibration, and softmax
        x = idx.index_select(0, input_pos).view(1, -1)
        llm_logits = model(x, max_seq_length, input_pos)[0, -1] / temperature_llm
        llm_probs = torch.nn.functional.softmax(llm_logits, dim=-1)
        llm_uc = - torch.dot(llm_probs, torch.log(llm_probs))

        #E.q(10) in paper, use calibrated prob for next token prediction
        prob = llm_probs + (sigmoid(llm_uc) - 0.5) * asr_prob

        idx_next = torch.max(prob, dim=-1).indices.to(dtype=dtype).to(device)

        # update sequence for asr and llm
        decoder_input = torch.cat((decoder_input, idx_next.unsqueeze(dim=0)), dim=1)

        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        if idx_next == eos_id:
            return idx[:input_pos]

    return idx


def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_file(), tokenizer_path

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint:
        name = llama_model_lookup(checkpoint)

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(name)

        model.load_state_dict(checkpoint)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
    prompt_length = encoded.size(0)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k)
        t = time.perf_counter() - t0

        model.reset_cache()
        print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


def mask_nan(x):
    nan_mask = torch.isnan(x)
    x[nan_mask] = 0
    return x

def compute_confidence(logits, T):
    logits = logits / T  # 32000
    probability = F.softmax(logits, dim=-1)
    confidence, _ = torch.max(probability, dim=-1)
    return 100 * confidence


def binary_search(logits, accuracy, threshold=1e-2):
    print(f"search for best T calibrated for accuracy: {accuracy:.2f}%")
    T_min = 0.1
    T_max = 10

    T_best = binary_search_recurse(logits, T_min, T_max, accuracy, threshold)

    return T_best

def binary_search_recurse(logits, T_min, T_max, accuracy, threshold=1e-1):

    conf_min = compute_confidence(logits.clone(), T_min)
    gap_min = abs(conf_min - accuracy)

    conf_max = compute_confidence(logits.clone(), T_max)
    gap_max = abs(conf_max - accuracy)

    print(f"=> T_left  {T_min:.4f} confidence {conf_min:.2f}% gap {gap_min:.2f}"
          f" T_right {T_max:.4f} confidence {conf_max:.2f}% gap {gap_max:.2f}")

    alpha = gap_min / (gap_min + gap_max)
    T_mid = T_min * (1 - alpha) + T_max * alpha
    conf_mid = compute_confidence(logits.clone(), T_mid)
    gap_mid = abs(conf_mid - accuracy)
    print(f"=> verifying T: {T_mid:.4f} confidence: {conf_mid:.2f}% gap: {gap_mid:.2f}")

    if gap_mid <= threshold:
        print(f"=> done! best T is {T_mid:.4f} with confidence: {conf_mid:.2f}")
        return T_mid

    if conf_max > accuracy:
        T_max *= 1.1
    elif conf_min < accuracy:
        T_min /= 1.1
    elif gap_min < gap_max:
        T_max = T_mid
    else:
        T_min = T_mid
    return binary_search_recurse(logits, T_min, T_max, accuracy, threshold)




if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore",
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)
