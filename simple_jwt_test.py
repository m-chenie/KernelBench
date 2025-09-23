#!/usr/bin/env python3
"""
Simple JWT authentication test for KernelBench - no user input required
"""

import os
import sys

# Add src to path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def simple_jwt_test():
    """Quick test of JWT token generation"""
    from utils import generate_jwt_token
    
    print("🔧 Testing JWT token generation...")
    
    # Set environment variables (same as your working curl)
    os.environ["JWT_SECRET"] = "yourmom"
    os.environ["JWT_TEAM_ID"] = "tenstorrent" 
    os.environ["JWT_TOKEN_ID"] = "debug-test"
    
    # Generate token
    token = generate_jwt_token()
    print(f"✅ Generated JWT token: {token}")
    
    # Show curl command that should work
    print(f"\n📋 Your working curl command equivalent:")
    print(f"export BEARER_TOKEN='{token}'")
    print("# Set your BASE_URL, then run:")
    print("""curl -s --no-buffer -X POST "$BASE_URL/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $BEARER_TOKEN" \\
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "messages": [{"role":"user","content":"Say hello and name one Tenstorrent product."}],
    "max_tokens": 128,
    "temperature": 0.2
  }' | jq -r '.choices[0].message.content // .error?.message // .'""")
    
    return token

if __name__ == "__main__":
    print("KernelBench JWT Integration - Quick Test")
    print("=" * 45)
    
    token = simple_jwt_test()
    
    print(f"\n🎉 JWT integration is working!")
    print(f"\n📝 To use in your KernelBench code:")
    print(f"   1. Set environment variables:")
    print(f"      export JWT_SECRET='yourmom'")
    print(f"      export JWT_BASE_URL='your-server-url'")
    print(f"      export JWT_TEAM_ID='tenstorrent'")
    print(f"      export JWT_TOKEN_ID='debug-test'")
    print(f"   2. Use server_type='jwt' in query_server calls")
    print(f"\n✨ Ready to use JWT authentication in KernelBench!")