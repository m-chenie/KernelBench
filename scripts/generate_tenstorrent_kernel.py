import os, sys
import torch
import json
import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Add the parent directory to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompt_constructor import prompt_generate_ttmetal_kernel_from_examples, prompt_generate_ttmetal_kernel_simple
from src.utils import extract_first_code, read_file, create_inference_server_from_presets

"""
Generate and evaluate Tenstorrent TT-Metal kernels for TT-Metal specific problems

This script generates C++ TT-Metal kernels for problems designed specifically for 
Tenstorrent hardware characteristics, rather than adapting CUDA-focused problems.
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

# TT-Metal specific problems designed for Tenstorrent hardware
TTMETAL_PROBLEMS = {
    # Level 1: Basic single-core operations
    "elementwise_add": {
        "name": "Element-wise Addition",
        "description": "Implement element-wise addition of two matrices using single core",
        "level": 1,
        "details": "Input: Two MxN matrices A and B. Output: Matrix C where C[i][j] = A[i][j] + B[i][j]. Use single core, proper tiling, and DRAM buffers."
    },
    "relu_activation": {
        "name": "ReLU Activation", 
        "description": "Apply ReLU activation function to a matrix using single core",
        "level": 1,
        "details": "Input: MxN matrix A. Output: Matrix B where B[i][j] = max(0, A[i][j]). Focus on compute kernel implementation."
    },
    "matrix_transpose": {
        "name": "Matrix Transpose",
        "description": "Transpose a matrix using single core with efficient memory access",
        "level": 1, 
        "details": "Input: MxN matrix A. Output: NxM matrix B where B[j][i] = A[i][j]. Demonstrate different memory access patterns."
    },
    
    # Level 2: Multi-core extensions
    "multicore_matmul": {
        "name": "Multi-core Matrix Multiplication",
        "description": "Extend single-core matmul to use multiple cores with work distribution",
        "level": 2,
        "details": "Based on the single-core matmul example, parallelize across multiple cores. Implement proper work distribution and inter-core coordination."
    },
    "multicore_elementwise": {
        "name": "Multi-core Element-wise Operations",
        "description": "Distribute element-wise operations across multiple cores",
        "level": 2,
        "details": "Take element-wise addition and parallelize it across cores. Teach work distribution for simpler operations."
    },
    
    # Level 3: Advanced patterns and optimizations
    "fused_matmul_relu": {
        "name": "Fused MatMul + ReLU",
        "description": "Combine matrix multiplication with ReLU activation in single kernel",
        "level": 3,
        "details": "Optimize by fusing matmul and ReLU to avoid intermediate DRAM writes. Show kernel fusion techniques."
    },
    "reduction_sum": {
        "name": "Matrix Reduction (Sum)",
        "description": "Implement tree reduction to sum all elements in a matrix",
        "level": 3,
        "details": "Different computation pattern teaching tree reductions. Important building block for many algorithms."
    },
    "conv2d_basic": {
        "name": "Basic 2D Convolution",
        "description": "Implement basic 2D convolution operation",
        "level": 3,
        "details": "More complex memory patterns and compute. Builds toward realistic ML workloads with sliding window operations."
    }
}

@dataclass
class TenstorrentConfig:
    # Problem selection
    problem_name: str = None
    
    # Generation configuration  
    server_type: str = "openai"
    model_name: str = "gpt-4-turbo"  # Changed to GPT-4 Turbo for larger context
    temperature: float = 0.8
    max_tokens: int = 4096  # Can increase this now with larger context
    
    # Tenstorrent specific configuration
    device_id: int = 0
    use_detailed_hardware_info: bool = True
    
    # Logging and output
    log_prompt: bool = True
    log_generated_kernel: bool = True
    verbose: bool = True
    logdir: str = "logs/tenstorrent"

@dataclass
class TenstorrentWorkArgs:
    problem_name: str
    sample_id: int
    device_id: int

@dataclass
class KernelExecutionResult:
    """Results from executing a TT-Metal kernel"""
    success: bool
    compilation_success: bool
    execution_success: bool
    compilation_time: float
    execution_time: float
    stdout: str
    stderr: str
    return_code: int
    error_message: str = ""

def compile_ttmetal_kernel(kernel_path: str, config: TenstorrentConfig) -> tuple[bool, str, str, float]:
    """
    Compile the TT-Metal kernel using the proper TT-Metal build system
    """
    start_time = time.time()
    try:
        # Get the TT-Metal environment
        tt_metal_home = "/home/m48chen/tt-metal"
        if not os.path.exists(tt_metal_home):
            return False, "", "TT-Metal installation not found", time.time() - start_time
        
        # Get absolute paths
        kernel_path_abs = os.path.abspath(kernel_path)
        kernel_name = Path(kernel_path).stem
        build_dir = Path(kernel_path).parent / f"{kernel_name}_build"
        build_dir.mkdir(exist_ok=True)
        
        # Create a CMakeLists.txt using the same pattern as TT-Metal examples
        cmake_content = f"""
cmake_minimum_required(VERSION 3.22...3.30)
project({kernel_name})

# Set the CMAKE_PREFIX_PATH to help find TT-Metalium
list(APPEND CMAKE_PREFIX_PATH {tt_metal_home}/build)

add_executable({kernel_name})
target_sources({kernel_name} PRIVATE {kernel_path_abs})

# Find and link TT-Metalium just like the official examples
if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries({kernel_name} PUBLIC TT::Metalium)
"""
        
        cmake_file = build_dir / "CMakeLists.txt"
        with open(cmake_file, 'w') as f:
            f.write(cmake_content)
        
        # Build using cmake with the same approach as TT-Metal build system
        compile_cmd = f"""
cd {build_dir} && 
CMAKE_PREFIX_PATH={tt_metal_home}/build cmake . -DCMAKE_BUILD_TYPE=Release && 
make -j$(nproc)
"""
        
        result = subprocess.run(
            compile_cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        compilation_time = time.time() - start_time
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr, compilation_time
        else:
            return False, result.stdout, result.stderr, compilation_time
            
    except subprocess.TimeoutExpired:
        return False, "", "Compilation timeout after 5 minutes", time.time() - start_time
    except Exception as e:
        return False, "", f"Compilation error: {str(e)}", time.time() - start_time

def compile_ttmetal_kernel_simple(kernel_path: str, config: TenstorrentConfig) -> tuple[bool, str, str, float]:
    """
    Try to compile the TT-Metal kernel using a simpler approach based on the existing build system
    """
    start_time = time.time()
    try:
        # Get the TT-Metal environment
        tt_metal_home = "/home/m48chen/tt-metal"
        if not os.path.exists(tt_metal_home):
            return False, "", "TT-Metal installation not found", time.time() - start_time
        
        # Get absolute paths
        kernel_path_abs = os.path.abspath(kernel_path)
        kernel_name = Path(kernel_path).stem
        build_dir = Path(kernel_path).parent / f"{kernel_name}_simple_build"
        build_dir.mkdir(exist_ok=True)
        
        # Try to compile directly using the includes and libraries we can find
        compile_cmd = f"""
cd {build_dir} &&
g++ -std=c++17 \\
    -I{tt_metal_home}/tt_metal \\
    -I{tt_metal_home}/build/include \\
    -I{tt_metal_home}/tt_metal/include \\
    -I{tt_metal_home}/tt_metal/third_party/fmt/include \\
    -L{tt_metal_home}/build/lib \\
    -o {kernel_name} {kernel_path_abs} \\
    -ltt_metal -ldevice -lfmt -pthread -ldl \\
    -DFMT_HEADER_ONLY=1 \\
    -Wno-unused-parameter
"""
        
        result = subprocess.run(
            compile_cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        compilation_time = time.time() - start_time
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr, compilation_time
        else:
            return False, result.stdout, result.stderr, compilation_time
            
    except subprocess.TimeoutExpired:
        return False, "", "Compilation timeout after 5 minutes", time.time() - start_time
    except Exception as e:
        return False, "", f"Compilation error: {str(e)}", time.time() - start_time

def execute_ttmetal_kernel_advanced(kernel_path: str, config: TenstorrentConfig) -> KernelExecutionResult:
    """
    Execute the generated TT-Metal kernel with proper error handling and logging
    """
    kernel_name = Path(kernel_path).stem
    build_dir = Path(kernel_path).parent / f"{kernel_name}_build"
    executable_path = build_dir / kernel_name
    
    # First, compile the kernel
    print("🔨 Compiling TT-Metal kernel...")
    comp_success, comp_stdout, comp_stderr, comp_time = compile_ttmetal_kernel(kernel_path, config)
    
    if not comp_success:
        return KernelExecutionResult(
            success=False,
            compilation_success=False,
            execution_success=False,
            compilation_time=comp_time,
            execution_time=0.0,
            stdout=comp_stdout,
            stderr=comp_stderr,
            return_code=-1,
            error_message="Compilation failed"
        )
    
    print(f"✅ Compilation successful in {comp_time:.2f} seconds")
    
    # Execute the compiled kernel
    print("🚀 Executing TT-Metal kernel...")
    start_time = time.time()
    
    try:
        # Set up TT-Metal environment variables
        env = os.environ.copy()
        env.update({
            "TT_METAL_HOME": "/home/m48chen/tt-metal",
            "ARCH_NAME": "wormhole_b0",
            "LD_LIBRARY_PATH": f"/home/m48chen/tt-metal/build/lib:{env.get('LD_LIBRARY_PATH', '')}"
        })
        
        result = subprocess.run(
            str(executable_path), 
            capture_output=True, 
            text=True,
            timeout=60,  # 1 minute timeout
            env=env
        )
        
        execution_time = time.time() - start_time
        
        return KernelExecutionResult(
            success=result.returncode == 0,
            compilation_success=True,
            execution_success=result.returncode == 0,
            compilation_time=comp_time,
            execution_time=execution_time,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            error_message="" if result.returncode == 0 else "Execution failed"
        )
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return KernelExecutionResult(
            success=False,
            compilation_success=True,
            execution_success=False,
            compilation_time=comp_time,
            execution_time=execution_time,
            stdout="",
            stderr="Execution timeout after 1 minute",
            return_code=-1,
            error_message="Execution timeout"
        )
    except Exception as e:
        execution_time = time.time() - start_time
        return KernelExecutionResult(
            success=False,
            compilation_success=True,
            execution_success=False,
            compilation_time=comp_time,
            execution_time=execution_time,
            stdout="",
            stderr=str(e),
            return_code=-1,
            error_message=f"Execution error: {str(e)}"
        )

def execute_ttmetal_kernel_simple(kernel_path: str, config: TenstorrentConfig) -> KernelExecutionResult:
    """
    Execute the generated TT-Metal kernel using the simpler compilation approach
    """
    kernel_name = Path(kernel_path).stem
    build_dir = Path(kernel_path).parent / f"{kernel_name}_simple_build"
    executable_path = build_dir / kernel_name
    
    # First, compile the kernel using the simple approach
    print("🔨 Compiling TT-Metal kernel (simple approach)...")
    comp_success, comp_stdout, comp_stderr, comp_time = compile_ttmetal_kernel_simple(kernel_path, config)
    
    if not comp_success:
        return KernelExecutionResult(
            success=False,
            compilation_success=False,
            execution_success=False,
            compilation_time=comp_time,
            execution_time=0.0,
            stdout=comp_stdout,
            stderr=comp_stderr,
            return_code=-1,
            error_message="Compilation failed"
        )
    
    print(f"✅ Compilation successful in {comp_time:.2f} seconds")
    
    # Execute the compiled kernel
    print("🚀 Executing TT-Metal kernel...")
    start_time = time.time()
    
    try:
        # Set up TT-Metal environment variables
        env = os.environ.copy()
        env.update({
            "TT_METAL_HOME": "/home/m48chen/tt-metal",
            "ARCH_NAME": "wormhole_b0",
            "LD_LIBRARY_PATH": f"/home/m48chen/tt-metal/build/lib:{env.get('LD_LIBRARY_PATH', '')}"
        })
        
        result = subprocess.run(
            str(executable_path), 
            capture_output=True, 
            text=True,
            timeout=60,  # 1 minute timeout
            env=env
        )
        
        execution_time = time.time() - start_time
        
        return KernelExecutionResult(
            success=result.returncode == 0,
            compilation_success=True,
            execution_success=result.returncode == 0,
            compilation_time=comp_time,
            execution_time=execution_time,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            error_message="" if result.returncode == 0 else "Execution failed"
        )
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return KernelExecutionResult(
            success=False,
            compilation_success=True,
            execution_success=False,
            compilation_time=comp_time,
            execution_time=execution_time,
            stdout="",
            stderr="Execution timeout after 1 minute",
            return_code=-1,
            error_message="Execution timeout"
        )
    except Exception as e:
        execution_time = time.time() - start_time
        return KernelExecutionResult(
            success=False,
            compilation_success=True,
            execution_success=False,
            compilation_time=comp_time,
            execution_time=execution_time,
            stdout="",
            stderr=str(e),
            return_code=-1,
            error_message=f"Execution error: {str(e)}"
        )

def log_execution_results(result: KernelExecutionResult, kernel_path: str, problem_name: str, run_dir: str):
    """
    Log detailed execution results to a JSON file
    """
    results_log = {
        "problem_name": problem_name,
        "kernel_path": kernel_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "compilation": {
            "success": result.compilation_success,
            "time_seconds": result.compilation_time,
        },
        "execution": {
            "success": result.execution_success,
            "time_seconds": result.execution_time,
            "return_code": result.return_code,
        },
        "output": {
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
        "overall_success": result.success,
        "error_message": result.error_message
    }
    
    # Save to JSON file
    results_file = os.path.join(run_dir, f"{problem_name}_execution_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_log, f, indent=2)
    
    print(f"📊 Execution results logged to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("🔍 EXECUTION SUMMARY")
    print("="*60)
    print(f"Problem: {problem_name}")
    print(f"Compilation: {'✅ SUCCESS' if result.compilation_success else '❌ FAILED'} ({result.compilation_time:.2f}s)")
    print(f"Execution: {'✅ SUCCESS' if result.execution_success else '❌ FAILED'} ({result.execution_time:.2f}s)")
    print(f"Overall: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    
    if result.stdout:
        print(f"\n📤 STDOUT:\n{result.stdout}")
    
    if result.stderr:
        print(f"\n📤 STDERR:\n{result.stderr}")
        
    if result.error_message:
        print(f"\n❌ ERROR: {result.error_message}")
    
    print("="*60)

def analyze_generated_kernel_quality(kernel_path: str, problem_spec: dict) -> dict:
    """
    Analyze the quality and correctness of generated TT-Metal kernel without compilation
    """
    with open(kernel_path, 'r') as f:
        kernel_code = f.read()
    
    analysis = {
        "syntax_checks": {},
        "api_usage": {},
        "best_practices": {},
        "problem_specific": {},
        "overall_score": 0
    }
    
    # Syntax and structure checks
    analysis["syntax_checks"]["has_includes"] = "#include" in kernel_code
    analysis["syntax_checks"]["has_main"] = "int main(" in kernel_code
    analysis["syntax_checks"]["balanced_braces"] = kernel_code.count('{') == kernel_code.count('}')
    analysis["syntax_checks"]["has_namespace"] = "using namespace" in kernel_code
    
    # TT-Metal API usage checks
    api_patterns = [
        "CreateDevice", "CloseDevice", "CreateProgram", "CoreCoord",
        "CreateKernel", "EnqueueProgram", "InterleavedBufferConfig",
        "CreateBuffer", "EnqueueWriteBuffer", "EnqueueReadBuffer"
    ]
    
    for pattern in api_patterns:
        analysis["api_usage"][f"uses_{pattern}"] = pattern in kernel_code
    
    # Best practices checks
    analysis["best_practices"]["uses_bfloat16"] = "bfloat16" in kernel_code
    analysis["best_practices"]["handles_tiling"] = "TILE_HEIGHT" in kernel_code or "TILE_WIDTH" in kernel_code
    analysis["best_practices"]["error_handling"] = "try" in kernel_code and "catch" in kernel_code
    analysis["best_practices"]["proper_cleanup"] = "CloseDevice" in kernel_code
    
    # Problem-specific checks for element-wise addition
    if problem_spec["name"] == "Element-wise Addition":
        analysis["problem_specific"]["two_input_matrices"] = kernel_code.count("vector<bfloat16>") >= 2
        analysis["problem_specific"]["element_wise_operation"] = "+" in kernel_code or "add" in kernel_code.lower()
        analysis["problem_specific"]["single_core"] = "CoreCoord core({0, 0})" in kernel_code
    
    # Calculate overall score
    total_checks = 0
    passed_checks = 0
    
    for category in analysis.values():
        if isinstance(category, dict):
            for check, result in category.items():
                total_checks += 1
                if result:
                    passed_checks += 1
    
    analysis["overall_score"] = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    return analysis

def generate_kernel_report(kernel_path: str, problem_name: str, run_dir: str):
    """
    Generate a comprehensive report about the generated kernel
    """
    problem_spec = TTMETAL_PROBLEMS[problem_name]
    analysis = analyze_generated_kernel_quality(kernel_path, problem_spec)
    
    # Create detailed report
    report = {
        "problem_name": problem_name,
        "problem_level": problem_spec["level"],
        "kernel_path": kernel_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis": analysis,
        "recommendations": []
    }
    
    # Generate recommendations based on analysis
    if not analysis["api_usage"]["uses_CreateDevice"]:
        report["recommendations"].append("Add proper device initialization with CreateDevice()")
    
    if not analysis["best_practices"]["uses_bfloat16"]:
        report["recommendations"].append("Use bfloat16 data type for optimal Tenstorrent performance")
    
    if not analysis["best_practices"]["handles_tiling"]:
        report["recommendations"].append("Implement proper 32x32 tiling for Tenstorrent hardware")
    
    if analysis["overall_score"] < 70:
        report["recommendations"].append("Consider regenerating with different prompt or temperature")
    
    # Save report
    report_file = os.path.join(run_dir, f"{problem_name}_quality_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 KERNEL QUALITY ANALYSIS")
    print("="*60)
    print(f"Problem: {problem_name} (Level {problem_spec['level']})")
    print(f"Overall Score: {analysis['overall_score']:.1f}%")
    
    print("\n🔍 API Usage:")
    for check, result in analysis["api_usage"].items():
        status = "✅" if result else "❌"
        print(f"  {status} {check.replace('uses_', '').replace('_', ' ').title()}")
    
    print("\n🏆 Best Practices:")
    for check, result in analysis["best_practices"].items():
        status = "✅" if result else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    if analysis["problem_specific"]:
        print(f"\n🎯 Problem-Specific ({problem_spec['name']}):")
        for check, result in analysis["problem_specific"].items():
            status = "✅" if result else "❌"
            print(f"  {status} {check.replace('_', ' ').title()}")
    
    if report["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
    
    print(f"\n📊 Full report saved: {report_file}")
    print("="*60)

def list_available_problems():
    """List all available TT-Metal problems"""
    print("Available TT-Metal Problems:")
    print("=" * 50)
    
    for level in [1, 2, 3]:
        level_problems = [(name, spec) for name, spec in TTMETAL_PROBLEMS.items() if spec['level'] == level]
        if level_problems:
            print(f"\nLevel {level}:")
            for name, spec in level_problems:
                print(f"  {name}: {spec['description']}")

def test_generated_ttmetal_kernel(kernel_path: str, problem_spec: dict, config: TenstorrentConfig) -> bool:
    """
    Basic test to see if the generated TT-Metal C++ kernel has proper structure
    """
    try:
        # Read the generated kernel
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()
        
        # Check for essential TT-Metal API components
        required_patterns = [
            "#include",  # Should have includes
            "CreateDevice",  # Device initialization
            "CreateProgram",  # Program creation
            "CoreCoord",  # Core specification
            "CreateKernel",  # Kernel creation
            "EnqueueProgram",  # Program execution
            "CloseDevice"  # Cleanup
        ]
        
        # Additional checks based on problem level
        if problem_spec['level'] >= 2:
            required_patterns.extend(["CoreRange", "CoreGrid"])  # Multi-core patterns
        
        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in kernel_code:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            print(f"❌ Generated kernel missing required patterns: {missing_patterns}")
            return False
        
        print("✅ Generated TT-Metal kernel contains required API patterns")
        
        # Check if it looks like valid C++ structure
        if kernel_code.count('{') != kernel_code.count('}'):
            print("❌ Generated kernel has mismatched braces")
            return False
            
        if "int main()" not in kernel_code and "main(" not in kernel_code:
            print("⚠️  Generated kernel might be missing main function")
            
        return True
            
    except Exception as e:
        print(f"❌ Error testing generated TT-Metal kernel: {e}")
        return False

def execute_ttmetal_kernel(kernel_path: str, config: TenstorrentConfig) -> bool:
    """
    Execute the generated TT-Metal kernel and check for runtime errors
    """
    try:
        # Compile the kernel
        compile_cmd = f"g++ -o {kernel_path}.out {kernel_path} -lttmetal"
        subprocess.run(compile_cmd, shell=True, check=True)
        
        # Execute the compiled kernel
        exec_cmd = f"./{kernel_path}.out"
        result = subprocess.run(exec_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Kernel execution failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
        
        print("✅ Kernel executed successfully")
        print(f"Output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during kernel execution: {e}")
        return False

def generate_tenstorrent_kernel_single(
    work: TenstorrentWorkArgs, 
    config: TenstorrentConfig, 
    inference_server: callable, 
    run_dir: str
) -> bool:
    """
    Generate a single Tenstorrent TT-Metal kernel for a given problem
    """
    problem_name = work.problem_name
    sample_id = work.sample_id
    
    # Get problem specification
    if problem_name not in TTMETAL_PROBLEMS:
        raise ValueError(f"Problem '{problem_name}' not found in TTMETAL_PROBLEMS")
    
    problem_spec = TTMETAL_PROBLEMS[problem_name]
    
    # Create detailed problem description for the LLM
    problem_description = f"""
Problem: {problem_spec['name']}
Level: {problem_spec['level']}
Description: {problem_spec['description']}

Detailed Requirements:
{problem_spec['details']}

Implementation Requirements:
- Use TT-Metal C++ API
- Target Tenstorrent Wormhole B0 architecture
- Handle data in 32x32 tiles (TILE_HEIGHT x TILE_WIDTH)
- Use bfloat16 data type
- Include proper error handling and cleanup
- Follow the programming pattern shown in the matrix multiplication example
"""

    # Generate prompt for Tenstorrent TT-Metal kernel
    if config.use_detailed_hardware_info:
        ttmetal_prompt = prompt_generate_ttmetal_kernel_from_examples(problem_description)
    else:
        ttmetal_prompt = prompt_generate_ttmetal_kernel_simple(problem_description)
    
    if config.log_prompt:
        prompt_path = os.path.join(run_dir, f"{problem_name}_sample_{sample_id}_ttmetal_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(ttmetal_prompt)
    
    # Query LLM for Tenstorrent kernel generation
    ttmetal_kernel = inference_server(ttmetal_prompt)
    ttmetal_kernel = extract_first_code(ttmetal_kernel, ["cpp", "c++"])
    
    # Check if LLM generated valid code
    assert ttmetal_kernel is not None, "Tenstorrent TT-Metal kernel generation failed"
    
    if config.verbose:
        print(f"Generated Tenstorrent TT-Metal kernel for problem: {problem_spec['name']}")
    
    # Store generated kernel
    kernel_path = os.path.join(run_dir, f"{problem_name}_sample_{sample_id}_ttmetal_kernel.cpp")
    with open(kernel_path, "w") as f:
        f.write(ttmetal_kernel)
    
    return True

def comprehensive_kernel_evaluation(kernel_path: str, problem_name: str, config: TenstorrentConfig, run_dir: str):
    """
    Comprehensive evaluation of the generated kernel including quality analysis and execution attempts
    """
    problem_spec = TTMETAL_PROBLEMS[problem_name]
    
    # 1. Quality Analysis (always works)
    print("📊 Analyzing kernel quality...")
    generate_kernel_report(kernel_path, problem_name, run_dir)
    
    # 2. Compilation Attempt (may fail)
    print("\n🔨 Attempting kernel compilation...")
    compilation_results = {
        "cmake_approach": None,
        "simple_approach": None
    }
    
    # Try CMake approach
    try:
        comp_success, comp_stdout, comp_stderr, comp_time = compile_ttmetal_kernel(kernel_path, config)
        compilation_results["cmake_approach"] = {
            "success": comp_success,
            "stdout": comp_stdout,
            "stderr": comp_stderr,
            "time": comp_time
        }
        if comp_success:
            print("✅ CMake compilation successful!")
        else:
            print("❌ CMake compilation failed")
    except Exception as e:
        compilation_results["cmake_approach"] = {
            "success": False,
            "error": str(e),
            "time": 0
        }
        print(f"❌ CMake compilation error: {e}")
    
    # Try simple g++ approach
    try:
        comp_success, comp_stdout, comp_stderr, comp_time = compile_ttmetal_kernel_simple(kernel_path, config)
        compilation_results["simple_approach"] = {
            "success": comp_success,
            "stdout": comp_stdout,
            "stderr": comp_stderr,
            "time": comp_time
        }
        if comp_success:
            print("✅ Simple g++ compilation successful!")
        else:
            print("❌ Simple g++ compilation failed")
    except Exception as e:
        compilation_results["simple_approach"] = {
            "success": False,
            "error": str(e),
            "time": 0
        }
        print(f"❌ Simple g++ compilation error: {e}")
    
    # 3. Execution Attempt (if any compilation succeeded)
    execution_results = None
    if compilation_results["cmake_approach"] and compilation_results["cmake_approach"]["success"]:
        print("\n🚀 Attempting kernel execution (CMake version)...")
        execution_results = execute_ttmetal_kernel_advanced(kernel_path, config)
    elif compilation_results["simple_approach"] and compilation_results["simple_approach"]["success"]:
        print("\n🚀 Attempting kernel execution (Simple version)...")
        execution_results = execute_ttmetal_kernel_simple(kernel_path, config)
    else:
        print("\n⚠️ Skipping execution - no successful compilation")
        execution_results = KernelExecutionResult(
            success=False,
            compilation_success=False,
            execution_success=False,
            compilation_time=0,
            execution_time=0,
            stdout="",
            stderr="No successful compilation",
            return_code=-1,
            error_message="Compilation failed"
        )
    
    # 4. Comprehensive Results Logging
    comprehensive_results = {
        "problem_name": problem_name,
        "problem_level": problem_spec["level"],
        "kernel_path": kernel_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": config.temperature,
        "model_name": config.model_name,
        "compilation_attempts": compilation_results,
        "execution_results": {
            "success": execution_results.success,
            "compilation_success": execution_results.compilation_success,
            "execution_success": execution_results.execution_success,
            "compilation_time": execution_results.compilation_time,
            "execution_time": execution_results.execution_time,
            "stdout": execution_results.stdout,
            "stderr": execution_results.stderr,
            "return_code": execution_results.return_code,
            "error_message": execution_results.error_message
        } if execution_results else None
    }
    
    # Save comprehensive results
    results_file = os.path.join(run_dir, f"{problem_name}_comprehensive_results.json")
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n📊 Comprehensive results saved to: {results_file}")
    
    # Print final summary
    print("\n" + "="*70)
    print("🎯 FINAL EVALUATION SUMMARY")
    print("="*70)
    print(f"Problem: {problem_spec['name']} (Level {problem_spec['level']})")
    print(f"Temperature: {config.temperature}")
    print(f"Model: {config.model_name}")
    
    cmake_status = "✅" if compilation_results.get("cmake_approach", {}).get("success") else "❌"
    simple_status = "✅" if compilation_results.get("simple_approach", {}).get("success") else "❌"
    exec_status = "✅" if execution_results and execution_results.success else "❌"
    
    print(f"CMake Compilation: {cmake_status}")
    print(f"Simple Compilation: {simple_status}")
    print(f"Execution: {exec_status}")
    
    if execution_results and execution_results.success:
        print("\n🎉 SUCCESS: Kernel generated, compiled, and executed successfully!")
    elif any(r.get("success") for r in compilation_results.values() if r):
        print("\n🔶 PARTIAL SUCCESS: Kernel generated and compiled, but execution failed")
    else:
        print("\n🔴 LIMITED SUCCESS: Kernel generated but compilation failed")
    
    print("="*70)
    
    return comprehensive_results

def main():
    """
    Generate Tenstorrent TT-Metal kernel for a specific problem
    """
    parser = argparse.ArgumentParser(description="Generate TT-Metal kernels for Tenstorrent hardware")
    parser.add_argument("--problem_name", type=str, required=True, help="Name of the problem to solve")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for LLM generation (0.0-1.0)")
    parser.add_argument("--device_id", type=int, default=0, help="Tenstorrent device ID")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo", help="LLM model to use")
    parser.add_argument("--server_type", type=str, default="openai", help="LLM server type")
    parser.add_argument("--logdir", type=str, default="logs/tenstorrent", help="Directory to save logs and generated kernels")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    config = TenstorrentConfig(
        problem_name=args.problem_name,
        temperature=args.temperature,
        device_id=args.device_id,
        model_name=args.model_name,
        server_type=args.server_type,
        logdir=args.logdir,
        verbose=args.verbose
    )
    
    # Show available problems if requested
    if config.problem_name == "list":
        list_available_problems()
        return
    
    # Validate problem name
    if config.problem_name not in TTMETAL_PROBLEMS:
        print(f"❌ Unknown problem: {config.problem_name}")
        print("\nUse '--problem_name=list' to see available problems")
        return
    
    problem_spec = TTMETAL_PROBLEMS[config.problem_name]
    print(f"🚀 Generating TT-Metal kernel for: {problem_spec['name']}")
    print(f"📋 Level {problem_spec['level']}: {problem_spec['description']}")
    print(f"🌡️  Temperature: {config.temperature}")
    print(f"🤖 Model: {config.model_name}")
    
    # Setup logging directory
    os.makedirs(config.logdir, exist_ok=True)
    
    # Create inference server
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        time_generation=True
    )
    
    # Setup run directory
    run_dir = os.path.join(config.logdir, "generated_kernels")
    os.makedirs(run_dir, exist_ok=True)
    
    # Generate kernel
    work = TenstorrentWorkArgs(
        problem_name=config.problem_name,
        sample_id=0,  # Single sample for now
        device_id=config.device_id
    )
    
    success = generate_tenstorrent_kernel_single(
        work, config, inference_server, run_dir
    )
    
    if success:
        print(f"✅ Successfully generated TT-Metal kernel for: {problem_spec['name']}")
        
        # Test the generated kernel
        kernel_path = os.path.join(run_dir, f"{config.problem_name}_sample_0_ttmetal_kernel.cpp")
        test_result = test_generated_ttmetal_kernel(kernel_path, problem_spec, config)
        
        if test_result:
            print("🎉 Generated kernel passed basic tests!")
            
            # Comprehensive evaluation
            comprehensive_kernel_evaluation(kernel_path, config.problem_name, config, run_dir)
            
        else:
            print("⚠️  Generated kernel failed basic tests")
            
            # Still run comprehensive evaluation even if basic tests fail
            comprehensive_kernel_evaluation(kernel_path, config.problem_name, config, run_dir)
        
        print(f"📁 Generated files saved in: {run_dir}")
        print(f"🔍 Kernel file: {kernel_path}")
        
    else:
        print(f"❌ Failed to generate TT-Metal kernel for: {problem_spec['name']}")

if __name__ == "__main__":
    main()