#!/usr/bin/env python3
"""
GPT-4 Kernel Generator for TT-Metal
Uses RAG properly: All context in system prompt, minimal user prompts.
Generates: compute kernel, reader kernel, writer kernel (no host code).
"""

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI
from datetime import datetime
import re

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Supported operations
OPERATIONS = {
    "subtract": {
        "description": "subtraction (C = A - B)",
        "kernel_name": "subtract_2_tiles",
        "operation_symbol": "-",
    },
    "multiply": {
        "description": "multiplication (C = A * B)",
        "kernel_name": "multiply_2_tiles",
        "operation_symbol": "*",
    },
}

def load_file(filepath):
    """Load a file and return its contents"""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return ""

def build_rag_knowledge_base():
    """Build the complete RAG knowledge base - all context in one place"""
    tt_metal_home = Path("/home/m48chen/tt-metal")
    
    # Load the complete addition example
    add_compute = load_file(
        tt_metal_home / "tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp"
    )
    add_reader = load_file(
        tt_metal_home / "tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp"
    )
    add_writer = load_file(
        tt_metal_home / "tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp"
    )
    
    # Load API headers (just the important parts)
    eltwise_binary = load_file(tt_metal_home / "tt_metal/include/compute_kernel_api/eltwise_binary.h")
    cb_api = load_file(tt_metal_home / "tt_metal/include/compute_kernel_api/cb_api.h")
    
    # Extract relevant API function signatures
    api_functions = extract_key_apis(eltwise_binary, cb_api)
    
    return {
        "api_functions": api_functions,
        "add_compute": add_compute,
        "add_reader": add_reader,
        "add_writer": add_writer,
    }

def extract_key_apis(eltwise_binary, cb_api):
    """Extract just the key API function signatures"""
    apis = []
    
    # Extract eltwise binary operations
    patterns = [
        r'ALWI void (add|sub|mul)_tiles_init\([^)]+\)',
        r'ALWI void (add|sub|mul)_tiles\([^)]+\)',
        r'ALWI void binary_op_init_common\([^)]+\)',
        r'ALWI void tile_regs_acquire\([^)]*\)',
        r'ALWI void tile_regs_commit\([^)]*\)',
        r'ALWI void tile_regs_wait\([^)]*\)',
        r'ALWI void tile_regs_release\([^)]*\)',
        r'ALWI void pack_tile\([^)]+\)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, eltwise_binary + cb_api)
        apis.extend(matches)
    
    # Extract circular buffer operations
    cb_patterns = [
        r'(cb_wait_front|cb_reserve_back|cb_push_back|cb_pop_front)\([^)]+\)'
    ]
    
    for pattern in cb_patterns:
        matches = re.findall(pattern, cb_api)
        apis.extend(matches)
    
    return "\n".join(set(apis)) if apis else ""

def create_system_prompt_with_rag(knowledge_base):
    """Create a rich system prompt with all RAG context"""
    return f"""You are an expert TT-Metal kernel developer for Tenstorrent's Wormhole architecture.

# KNOWLEDGE BASE (RAG Context)

## Available TT-Metal APIs

### Element-wise Binary Operations (from eltwise_binary.h):
- add_tiles_init(cb_in0, cb_in1) - Initialize addition
- add_tiles(cb_in0, cb_in1, itile0, itile1, idst) - Perform addition
- sub_tiles_init(cb_in0, cb_in1) - Initialize subtraction  
- sub_tiles(cb_in0, cb_in1, itile0, itile1, idst) - Perform subtraction
- mul_tiles_init(cb_in0, cb_in1) - Initialize multiplication
- mul_tiles(cb_in0, cb_in1, itile0, itile1, idst) - Perform multiplication
- binary_op_init_common(cb_in0, cb_in1, cb_out) - Common initialization for all ops

### Tile Register Management:
- tile_regs_acquire() - Acquire destination registers
- tile_regs_commit() - Commit to packer
- tile_regs_wait() - Wait for packer
- tile_regs_release() - Release registers
- pack_tile(dst_index, cb_out) - Pack tile to circular buffer

### Circular Buffer Operations:
- cb_wait_front(cb_id, num_tiles) - Wait for input
- cb_reserve_back(cb_id, num_tiles) - Reserve output space
- cb_push_back(cb_id, num_tiles) - Push output
- cb_pop_front(cb_id, num_tiles) - Pop input

## Reference Implementation: Addition Example

### Compute Kernel (add_2_tiles.cpp):
```cpp
{knowledge_base['add_compute']}
```

### Reader Kernel (reader_binary_1_tile.cpp):
```cpp
{knowledge_base['add_reader']}
```

### Writer Kernel (writer_1_tile.cpp):
```cpp
{knowledge_base['add_writer']}
```

# YOUR ROLE

You generate TT-Metal kernels by:
1. Understanding the pattern from the addition example
2. Selecting the correct API functions for the requested operation
3. Creating your own implementation (don't copy verbatim)
4. Following TT-Metal best practices (proper headers, comments, error handling)

Generate clean, production-quality code."""

def generate_kernel(kernel_type, operation, operation_symbol, kernel_name, system_prompt, model="gpt-4o"):
    """Generate a single kernel file with minimal user prompt"""
    
    # Minimal, focused user prompts that reference the RAG context
    if kernel_type == "compute":
        user_prompt = f"""Generate a TT-Metal compute kernel for {operation} (C = A {operation_symbol} B).

Requirements:
- Look in the knowledge base and find the {operation} API functions
- Follow the same pattern as the addition compute kernel
- Use circular buffers: CB_0 (input A), CB_1 (input B), CB_16 (output)
- Include comments explaining each step
- Filename: kernels/compute/{kernel_name}.cpp

Generate only the .cpp code."""

    elif kernel_type == "reader":
        user_prompt = f"""Generate a TT-Metal reader kernel for the {operation} example.

Key insight: The reader is operation-agnostic (just reads data from DRAM).

Requirements:
- Follow the same pattern as the addition reader kernel
- Read two tiles from DRAM to CB_0 and CB_1
- Use NOC async reads with barriers
- Runtime args: arg 0 = src0 address, arg 1 = src1 address
- Filename: kernels/dataflow/reader_binary_1_tile.cpp

Generate only the .cpp code."""

    elif kernel_type == "writer":
        user_prompt = f"""Generate a TT-Metal writer kernel for the {operation} example.

Key insight: The writer is operation-agnostic (just writes results to DRAM).

Requirements:
- Follow the same pattern as the addition writer kernel  
- Write one tile from CB_16 to DRAM
- Use NOC async write with barrier
- Runtime arg: arg 0 = dst address
- Filename: kernels/dataflow/writer_1_tile.cpp

Generate only the .cpp code."""
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Low temperature for consistency
            max_tokens=3000
        )
        
        generated_code = response.choices[0].message.content
        
        # Extract code from markdown if present
        if "```cpp" in generated_code:
            generated_code = generated_code.split("```cpp")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
        return generated_code, {"user": user_prompt}
        
    except Exception as e:
        print(f"Error calling GPT-4 for {kernel_type}: {e}")
        return None, None

def save_prompts_file(output_dir, operation, model, all_prompts):
    """Save minimal prompt log"""
    prompts_file = output_dir / "GENERATION_PROMPTS.md"
    
    content = f"""# Kernel Generation Log

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Operation:** {operation}  
**Model:** {model}

## Note on RAG
The system prompt contains the complete knowledge base (API docs + addition example).
User prompts below are minimal - they just reference the RAG context.

---

"""
    
    for kernel_type, prompt_info in all_prompts.items():
        content += f"""## {kernel_type.upper()} Kernel

**User Prompt:**
```
{prompt_info['user']}
```

---

"""
    
    prompts_file.write_text(content)
    return prompts_file

def create_directory_structure(base_path):
    """Create kernel directory structure"""
    base = Path(base_path)
    base.mkdir(exist_ok=True)
    (base / "kernels" / "compute").mkdir(parents=True, exist_ok=True)
    (base / "kernels" / "dataflow").mkdir(parents=True, exist_ok=True)
    return base

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate TT-Metal kernels using GPT-4 with RAG",
        epilog="Example: %(prog)s --operation subtract"
    )
    
    parser.add_argument(
        "--operation",
        choices=list(OPERATIONS.keys()),
        required=True,
        help="Operation to generate kernels for"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ~/tt-metal/tt_metal/programming_examples/<op>_kernels_llm)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model (default: gpt-4o)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    op_config = OPERATIONS[args.operation]
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"/home/m48chen/tt-metal/tt_metal/programming_examples/{args.operation}_kernels_llm")
    
    print("="*80)
    print("TT-Metal Kernel Generator (RAG-based)")
    print("="*80)
    print(f"Operation: {op_config['description']}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set")
        return 1
    
    print("\n[1/5] Building RAG knowledge base...")
    knowledge_base = build_rag_knowledge_base()
    print("   ✓ Loaded addition example (compute, reader, writer)")
    print("   ✓ Extracted API documentation")
    
    print("\n[2/5] Creating system prompt with RAG context...")
    system_prompt = create_system_prompt_with_rag(knowledge_base)
    print(f"   ✓ System prompt ready ({len(system_prompt)} chars)")
    
    print(f"\n[3/5] Creating directory structure...")
    base_dir = create_directory_structure(output_dir)
    print(f"   ✓ Created: {base_dir}")
    
    # Generate the 3 kernels
    kernels_to_generate = [
        ("compute", f"kernels/compute/{op_config['kernel_name']}.cpp"),
        ("reader", "kernels/dataflow/reader_binary_1_tile.cpp"),
        ("writer", "kernels/dataflow/writer_1_tile.cpp"),
    ]
    
    all_prompts = {}
    
    print("\n[4/5] Generating kernels with GPT-4...")
    for kernel_type, filepath in kernels_to_generate:
        print(f"   Generating {kernel_type}...")
        
        generated_code, prompt_info = generate_kernel(
            kernel_type,
            op_config['description'],
            op_config['operation_symbol'],
            op_config['kernel_name'],
            system_prompt,
            model=args.model
        )
        
        if not generated_code:
            print(f"   ✗ Failed to generate {kernel_type}")
            return 1
        
        output_path = base_dir / filepath
        output_path.write_text(generated_code)
        all_prompts[kernel_type] = prompt_info
        
        print(f"   ✓ {kernel_type}: {len(generated_code)} bytes")
    
    print(f"\n[5/5] Saving generation log...")
    prompts_file = save_prompts_file(base_dir, op_config['description'], args.model, all_prompts)
    print(f"   ✓ Log saved: {prompts_file}")
    
    print("\n" + "="*80)
    print("✓ GENERATION COMPLETE")
    print("="*80)
    print(f"\nGenerated 3 kernels at: {output_dir}")
    print("\nFiles:")
    print(f"  • kernels/compute/{op_config['kernel_name']}.cpp")
    print(f"  • kernels/dataflow/reader_binary_1_tile.cpp")
    print(f"  • kernels/dataflow/writer_1_tile.cpp")
    print("\nNote: Host code and CMakeLists.txt not generated.")
    print("Use the addition example as a template for those.")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
