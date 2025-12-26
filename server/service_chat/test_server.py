"""
Test script để kiểm tra server có hoạt động không
Chạy: python test_server.py
"""
import requests
import sys

base_url = "http://localhost:8003"

print("Testing Chat API Server...")
print("=" * 50)

# Test 1: Root endpoint
print("\n1. Testing GET /")
try:
    response = requests.get(f"{base_url}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Upload info endpoint
print("\n2. Testing GET /upload")
try:
    response = requests.get(f"{base_url}/upload")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ Success!")
        print(f"   Response: {response.json()}")
    elif response.status_code == 405:
        print(f"   ❌ Method Not Allowed - Server is running OLD code!")
        print(f"   ⚠️  Please run: restart_server.bat")
        print(f"   Response: {response.text}")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Upload file
print("\n3. Testing POST /upload")
try:
    # Tạo test file
    test_content = b"Test file content"
    files = {"file": ("test.txt", test_content, "text/plain")}
    response = requests.post(f"{base_url}/upload", files=files)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ Upload successful!")
        print(f"   Response: {response.json()}")
    else:
        print(f"   ❌ Upload failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 50)
print("Test completed!")

