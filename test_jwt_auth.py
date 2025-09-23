#!/usr/bin/env python3
"""
Test script for JWT authentication integration in KernelBench
Demonstrates how to use the new JWT server type
"""

import os
import sys

# Add src to path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import query_server, create_inference_server_from_presets

def test_jwt_authentication():
    """Test JWT authentication with a simple query"""
    
    # Set up environment variables for JWT authentication
    # Update these based on your actual server configuration
    os.environ["JWT_SECRET"] = "yourmom"  # Your JWT secret
    os.environ["JWT_TEAM_ID"] = "tenstorrent"
    os.environ["JWT_TOKEN_ID"] = "debug-test"
    
    # Use the actual server - you'll need to set the correct BASE_URL
    # For example, if your server is running on port 8000 on tauceti:
    base_url = input("Enter your BASE_URL (e.g., http://tauceti.eng.uwaterloo.ca:8000): ").strip()
    if not base_url:
        base_url = "http://tauceti.eng.uwaterloo.ca:8000"  # Default assumption
    
    os.environ["JWT_BASE_URL"] = base_url
    
    print("Testing JWT authentication...")
    print(f"JWT_BASE_URL: {os.environ.get('JWT_BASE_URL')}")
    print(f"JWT_TEAM_ID: {os.environ.get('JWT_TEAM_ID')}")
    print(f"JWT_TOKEN_ID: {os.environ.get('JWT_TOKEN_ID')}")
    
    try:
        # Method 1: Direct query_server call
        print("\n🔄 Testing direct query_server call...")
        response = query_server(
            prompt="Say hello and name one Tenstorrent product.",
            server_type="jwt",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            temperature=0.2,
            max_tokens=128
        )
        
        print("✅ Direct query_server call successful!")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Direct query_server call failed: {e}")
        print("This might be expected if the server isn't running or URL is incorrect")
    
    try:
        # Method 2: Using preset configuration
        print("\n🔄 Testing preset-based query...")
        jwt_client = create_inference_server_from_presets(
            server_type="jwt",
            verbose=True
        )
        
        response = jwt_client("Say hello and name one Tenstorrent product.")
        
        print("✅ Preset-based query successful!")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Preset-based query failed: {e}")
        print("This might be expected if the server isn't running or URL is incorrect")

def test_jwt_token_generation():
    """Test the JWT token generation function"""
    from utils import generate_jwt_token
    
    print("Testing JWT token generation...")
    
    # Test with default values
    token1 = generate_jwt_token()
    print(f"Default token: {token1}")
    
    # Test with custom values
    token2 = generate_jwt_token(
        team_id="custom_team",
        token_id="custom_token",
        secret="custom_secret"
    )
    print(f"Custom token: {token2}")
    
    # Verify tokens are different
    assert token1 != token2, "Tokens should be different with different parameters"
    print("✅ JWT token generation working correctly!")

def test_manual_curl_equivalent():
    """Show how to manually test like your working curl command"""
    from utils import generate_jwt_token
    
    print("\n" + "="*50)
    print("MANUAL TESTING EQUIVALENT")
    print("="*50)
    
    # Generate token
    os.environ["JWT_SECRET"] = "yourmom"
    os.environ["JWT_TEAM_ID"] = "tenstorrent" 
    os.environ["JWT_TOKEN_ID"] = "debug-test"
    
    token = generate_jwt_token()
    base_url = os.environ.get("JWT_BASE_URL", "http://tauceti.eng.uwaterloo.ca:8000")
    
    print(f"Generated Bearer Token: {token}")
    print(f"Server URL: {base_url}")
    print("\nTo test manually with curl, run:")
    print(f"""
export BEARER_TOKEN="{token}"
export BASE_URL="{base_url}"

curl -s --no-buffer -X POST "$BASE_URL/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $BEARER_TOKEN" \\
  -d '{{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "messages": [{{"role":"user","content":"Say hello and name one Tenstorrent product."}}],
    "max_tokens": 128,
    "temperature": 0.2
  }}' | jq -r '.choices[0].message.content // .error?.message // .'
    """)

if __name__ == "__main__":
    print("KernelBench JWT Authentication Test")
    print("=" * 40)
    
    # Test JWT token generation first
    test_jwt_token_generation()
    
    # Show manual testing approach
    test_manual_curl_equivalent()
    
    # Test actual API calls 
    test_jwt_authentication()
    
    print("\n" + "=" * 40)
    print("Test completed!")
    print("\nTo use JWT authentication in your actual code:")
    print("1. Set the environment variables:")
    print("   export JWT_SECRET='your-secret'")
    print("   export JWT_BASE_URL='http://your-server:port'")
    print("   export JWT_TEAM_ID='your-team'")
    print("   export JWT_TOKEN_ID='your-token-id'")
    print("2. Use server_type='jwt' in your query_server calls")