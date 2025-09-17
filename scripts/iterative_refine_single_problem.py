import os
import json
import time
import datetime
import pydra
from pydra import Config, REQUIRED
import torch

from src.dataset import construct_kernelbench_dataset
from src.utils import (
    read_file,
    extract_first_code,
    create_inference_server_from_presets,
)
from src.prompt_constructor import (
    prompt_generate_custom_cuda_from_prompt_template,
)
from src.eval import eval_kernel_against_ref


REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IterativeConfig(Config):
    def __init__(self):
        # Problem selection
        self.dataset_src = REQUIRED  # 'local' or 'huggingface' (we assume local files here)
        self.level = 1
        self.problem_id = 1

        # Inference server
        self.server_type = "openai"
        self.model_name = "gpt-4o-2024-08-06"
        self.max_tokens = 4096

        # One-shot baseline
        self.oneshot_temperature = 0.0  # greedy baseline

        # Repeated sampling (Appendix C.2)
        self.k_samples = 5
        self.sample_temperature = 0.7
        # Optional comma-separated list of temperatures to sweep for independent samples (e.g., "0.1,0.2,0.4,0.7,1.0").
        # If provided, overrides k_samples and sample_temperature.
        self.sample_temps = ""

            # Iterative refinement (Appendix C.3)
        self.refine_rounds = 5
        # Use a lower temperature for refinement so the model can build on prior output deterministically.
        self.refine_temperature = 0.2
        # Choose refinement seed: 'best' (best of baseline+samples) or 'oneshot' (force baseline)
        # Default changed to 'oneshot' so refinement starts from baseline
        self.refine_seed = "oneshot"

        # Evaluation
        self.num_correct_trials = 5
        self.num_perf_trials = 100

        # Logging
        self.run_name = None  # default set in main
        self.out_dir = os.path.join(REPO_TOP_DIR, "runs")
        self.verbose_flag = False

        # GPU arch (auto-detect if None)
        self.gpu_arch = None

    def __repr__(self):
        return f"IterativeConfig({self.to_dict()})"

    # pydra method-tokens support: allow CLI ".verbose" or ".verbose_logging"
    def verbose(self):
        self.verbose_flag = True

    def verbose_logging(self):
        self.verbose_flag = True


def _auto_set_gpu_arch(verbose: bool = False):
    """Auto-detect current CUDA device arch and set TORCH_CUDA_ARCH_LIST accordingly.

    Torch's cpp_extension expects decimal SM strings like "7.5", not "75".
    Using the integer form can cause: ValueError("Unknown CUDA arch (75) or GPU not supported").
    """
    if not torch.cuda.is_available():
        print("[GPU ARCH] CUDA not available; continuing (evaluation will fail without GPU)")
        return None
    cc = torch.cuda.get_device_capability()
    major, minor = cc
    sm_str = f"{major}.{minor}"  # e.g., "7.5" for Turing (SM75)
    # Friendly arch names (for logs only)
    sm_name_map = {
        "7.5": "Turing",
        "8.0": "Ampere",
        "8.6": "Ampere",
        "8.9": "Ada",
        "9.0": "Hopper",
        "10.2": "Blackwell",
    }
    friendly = sm_name_map.get(sm_str, f"SM{major}{minor}")
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env_arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = sm_str
        if verbose:
            print(f"[GPU ARCH] Auto-set TORCH_CUDA_ARCH_LIST={sm_str} ({friendly})")
    else:
        # Normalize existing list like "7.5;8.0" and check presence
        existing = env_arch_list.replace(" ", "")
        if sm_str not in existing:
            print(
                f"[GPU ARCH][WARNING] Detected SM {sm_str} ({friendly}) not in existing TORCH_CUDA_ARCH_LIST='{env_arch_list}'.\n"
                f"Compilation may target the wrong architecture. Consider: export TORCH_CUDA_ARCH_LIST={sm_str}"
            )
    return sm_str


def _build_initial_prompt(ref_arch_src: str) -> str:
    """Paper C.1 one-shot baseline prompt via existing template util."""
    return prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)


def _build_iterative_prompt(initial_prompt: str, prev_generation: str, eval_feedback_text: str, runtime_ms: float | None = None, profiler_text: str | None = None) -> str:
    """
    Paper C.3 style prompt for an iterative turn:
    <Initial prompt>
    Here is your latest generation:
    <G>
    Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
    Here is your Evaluation Result:
    <Raw Compiler and Execution Feedback from stdout>
    <if correct>
    Your kernel executed successfully and produced the correct output.
    Here is your wall clock time: {runtime} milliseconds
    <Profiler information if used>
    Name your new improved output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
    """
    parts = [initial_prompt]
    parts.append("Here is your latest generation:\n" + prev_generation)
    parts.append(
        "Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model."
    )
    parts.append("Here is your Evaluation Result:\n" + eval_feedback_text)
    if runtime_ms is not None:
        parts.append("Your kernel executed successfully and produced the correct output.")
        parts.append(f"Here is your wall clock time: {runtime_ms} milliseconds")
    if profiler_text:
        parts.append(profiler_text)
    parts.append(
        "Name your new improved output architecture ModelNew. Output the new code in codeblocks. "
        "Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. "
        "Just output the new model code, no other text, and NO testing code!"
    )
    return "\n\n".join(parts)


def _format_eval_feedback(kernel_exec_result) -> tuple[str, float | None]:
    """Return (feedback_text, runtime_ms_if_correct)."""
    if kernel_exec_result is None:
        return ("Evaluation process encountered a lock or transient error. Please retry.", None)

    meta = kernel_exec_result.metadata or {}
    # Ensure JSON-serializable feedback
    try:
        feedback_json = json.dumps(meta, indent=2, default=str)
    except Exception:
        # Fallback: stringify
        feedback_json = json.dumps({k: str(v) for k, v in meta.items()}, indent=2)

    rt_ms = None
    if getattr(kernel_exec_result, "correctness", False) and kernel_exec_result.runtime_stats:
        # time_execution_with_cuda_event returns ms
        rt_ms = float(kernel_exec_result.runtime_stats.get("mean", -1))

    return (feedback_json, rt_ms)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_text(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)


def _write_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _evaluate(ref_arch_src: str, gen_code: str, cfg: IterativeConfig):
    return eval_kernel_against_ref(
        ref_arch_src,
        gen_code,
        verbose=cfg.verbose_flag,
        measure_performance=True,
        num_correct_trials=cfg.num_correct_trials,
        num_perf_trials=cfg.num_perf_trials,
    )


def _select_better(a, b):
    """Choose between two kernel results, preferring correctness then faster runtime."""
    if a is None:
        return b
    if b is None:
        return a
    a_ok = getattr(a, "correctness", False)
    b_ok = getattr(b, "correctness", False)
    if a_ok and not b_ok:
        return a
    if b_ok and not a_ok:
        return b
    if a_ok and b_ok:
        a_rt = a.runtime_stats.get("mean", float("inf")) if a.runtime_stats else float("inf")
        b_rt = b.runtime_stats.get("mean", float("inf")) if b.runtime_stats else float("inf")
        return a if a_rt <= b_rt else b
    # neither correct: prefer compiled True, else keep a
    a_comp = getattr(a, "compiled", False)
    b_comp = getattr(b, "compiled", False)
    if a_comp and not b_comp:
        return a
    if b_comp and not a_comp:
        return b
    return a


@pydra.main(base=IterativeConfig)
def main(cfg: IterativeConfig):
    print(f"Starting iterative baseline + sampling + refinement with cfg: {cfg}")

    # GPU arch
    _auto_set_gpu_arch(verbose=cfg.verbose_flag)

    # Dataset (local)
    assert cfg.dataset_src == "local", "This script currently assumes local dataset files."
    dataset = construct_kernelbench_dataset(cfg.level)
    assert 1 <= cfg.problem_id <= len(dataset), f"Problem ID {cfg.problem_id} out of range for level {cfg.level}"
    ref_path = dataset[cfg.problem_id - 1]
    ref_arch_src = read_file(ref_path)
    problem_name = os.path.basename(ref_path)

    # Inference servers
    oneshot_server = create_inference_server_from_presets(
        server_type=cfg.server_type,
        model_name=cfg.model_name,
        temperature=cfg.oneshot_temperature,
        max_tokens=cfg.max_tokens,
        verbose=cfg.verbose_flag,
        time_generation=True,
    )
    # Note: Sampling servers are created per-temperature below if a sweep is requested
    refine_server = create_inference_server_from_presets(
        server_type=cfg.server_type,
        model_name=cfg.model_name,
        temperature=cfg.refine_temperature,
        max_tokens=cfg.max_tokens,
        verbose=cfg.verbose_flag,
        time_generation=True,
    )

    # Output dirs
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"iterative_l{cfg.level}_p{cfg.problem_id}_{ts}"
    base_dir = os.path.join(cfg.out_dir, run_name, f"level_{cfg.level}", f"problem_{cfg.problem_id}")
    _ensure_dir(base_dir)
    rounds_dir = os.path.join(base_dir, "rounds")
    samples_dir = os.path.join(base_dir, "samples")
    _ensure_dir(rounds_dir)
    _ensure_dir(samples_dir)

    # 1) One-shot baseline (C.1)
    initial_prompt = _build_initial_prompt(ref_arch_src)
    _write_text(os.path.join(base_dir, "prompt_oneshot.txt"), initial_prompt)
    oneshot_raw = oneshot_server(initial_prompt)
    oneshot_code = extract_first_code(oneshot_raw, ["python", "cpp"]) or ""
    _write_text(os.path.join(base_dir, "baseline_generated.py"), oneshot_code)
    oneshot_eval = _evaluate(ref_arch_src, oneshot_code, cfg)
    _write_json(
        os.path.join(base_dir, "baseline_eval.json"),
        {
            "compiled": getattr(oneshot_eval, "compiled", False) if oneshot_eval else False,
            "correctness": getattr(oneshot_eval, "correctness", False) if oneshot_eval else False,
            "runtime_stats": getattr(oneshot_eval, "runtime_stats", {}) if oneshot_eval else {},
            "metadata": getattr(oneshot_eval, "metadata", {}) if oneshot_eval else {},
        },
    )

    best_eval = oneshot_eval
    best_code = oneshot_code
    best_stage = "oneshot"

    # 2) Repeated sampling with independent prompts across temperatures (C.2)
    temps_list = None
    if cfg.sample_temps:
        try:
            # Support str like "0.1,0.2" or with spaces/parentheses, and also list/tuple inputs from CLI parsing
            if isinstance(cfg.sample_temps, (list, tuple)):
                temps_list = [float(x) for x in cfg.sample_temps]
            elif isinstance(cfg.sample_temps, str):
                s = cfg.sample_temps.strip()
                # Trim enclosing parentheses/brackets if present
                if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
                    s = s[1:-1]
                temps_list = [float(t.strip()) for t in s.split(',') if t.strip() != ""]
            else:
                # Fallback attempt via stringification
                s = str(cfg.sample_temps).strip()
                if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
                    s = s[1:-1]
                temps_list = [float(t.strip()) for t in s.split(',') if t.strip() != ""]
        except Exception:
            print(f"[WARN] Unable to parse sample_temps='{cfg.sample_temps}', falling back to k_samples @ sample_temperature.")
            temps_list = None

    if temps_list is None:
        temps_list = [cfg.sample_temperature] * cfg.k_samples

    for k, temp in enumerate(temps_list):
        sample_path = os.path.join(samples_dir, f"sample_{k}")
        _ensure_dir(sample_path)
        _write_text(os.path.join(sample_path, "prompt.txt"), initial_prompt)
        # Create an inference server for this temperature
        sample_server_k = create_inference_server_from_presets(
            server_type=cfg.server_type,
            model_name=cfg.model_name,
            temperature=temp,
            max_tokens=cfg.max_tokens,
            verbose=cfg.verbose_flag,
            time_generation=True,
        )
        raw = sample_server_k(initial_prompt)
        code = extract_first_code(raw, ["python", "cpp"]) or ""
        _write_text(os.path.join(sample_path, "generated.py"), code)
        eval_res = _evaluate(ref_arch_src, code, cfg)
        _write_json(
            os.path.join(sample_path, "eval.json"),
            {
                "compiled": getattr(eval_res, "compiled", False) if eval_res else False,
                "correctness": getattr(eval_res, "correctness", False) if eval_res else False,
                "runtime_stats": getattr(eval_res, "runtime_stats", {}) if eval_res else {},
                "metadata": {
                    **(getattr(eval_res, "metadata", {}) or {}),
                    "sample_temperature": temp,
                } if eval_res else {"sample_temperature": temp},
            },
        )
        best_eval = _select_better(best_eval, eval_res)
        if best_eval is eval_res:
            best_code = code
            best_stage = f"sample_{k}"

    # 3) Iterative refinement (C.3)
    # Choose seed based on cfg.refine_seed
    if (cfg.refine_seed or "best").lower() == "oneshot":
        curr_code = oneshot_code
        curr_eval = oneshot_eval
        seed_stage = "oneshot"
    else:
        curr_code = best_code
        curr_eval = best_eval
        seed_stage = best_stage

    if cfg.verbose_flag:
        print(f"[Refine] Seeding refinement from: {seed_stage}")
    for r in range(cfg.refine_rounds):
        round_dir = os.path.join(rounds_dir, f"round_{r}")
        _ensure_dir(round_dir)

        feedback_text, runtime_ms = _format_eval_feedback(curr_eval)
        refine_prompt = _build_iterative_prompt(
            initial_prompt=initial_prompt,
            prev_generation=curr_code,
            eval_feedback_text=feedback_text,
            runtime_ms=runtime_ms,
        )
        _write_text(os.path.join(round_dir, "prompt.txt"), refine_prompt)
        refined_raw = refine_server(refine_prompt)
        refined_code = extract_first_code(refined_raw, ["python", "cpp"]) or ""
        _write_text(os.path.join(round_dir, "generated.py"), refined_code)
        refined_eval = _evaluate(ref_arch_src, refined_code, cfg)
        _write_json(
            os.path.join(round_dir, "eval.json"),
            {
                "compiled": getattr(refined_eval, "compiled", False) if refined_eval else False,
                "correctness": getattr(refined_eval, "correctness", False) if refined_eval else False,
                "runtime_stats": getattr(refined_eval, "runtime_stats", {}) if refined_eval else {},
                "metadata": getattr(refined_eval, "metadata", {}) if refined_eval else {},
            },
        )

        # Update trajectories
        curr_code, curr_eval = refined_code, refined_eval
        best_eval = _select_better(best_eval, refined_eval)
        if best_eval is refined_eval:
            best_code = refined_code
            best_stage = f"round_{r}"

    # Summary
    summary = {
        "problem": {
            "level": cfg.level,
            "problem_id": cfg.problem_id,
            "problem_name": problem_name,
        },
        "server": {
            "server_type": cfg.server_type,
            "model_name": cfg.model_name,
        },
        "stages": {
            "oneshot": {
                "compiled": getattr(oneshot_eval, "compiled", False) if oneshot_eval else False,
                "correctness": getattr(oneshot_eval, "correctness", False) if oneshot_eval else False,
                "runtime_stats": getattr(oneshot_eval, "runtime_stats", {}) if oneshot_eval else {},
            },
            "best_stage": best_stage,
            "best": {
                "compiled": getattr(best_eval, "compiled", False) if best_eval else False,
                "correctness": getattr(best_eval, "correctness", False) if best_eval else False,
                "runtime_stats": getattr(best_eval, "runtime_stats", {}) if best_eval else {},
            },
        },
    }
    _write_json(os.path.join(base_dir, "summary.json"), summary)
    print("\nCompleted. Best stage:", best_stage)
    print("Best stats:", summary["stages"]["best"])


if __name__ == "__main__":
    main()
