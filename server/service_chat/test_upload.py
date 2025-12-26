"""
Test script để kiểm tra upload endpoint
Chạy: python test_upload.py
"""
import requests

# Test upload endpoint
url = "http://localhost:8003/upload"

# Tạo file test
test_file_path = "test.txt"
with open(test_file_path, "w") as f:
    f.write("Test file content")

try:
    with open(test_file_path, "rb") as f:
        files = {"file": ("test.txt", f, "text/plain")}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ Upload endpoint hoạt động!")
    else:
        print("❌ Upload endpoint có vấn đề!")
        
except Exception as e:
    print(f"❌ Lỗi: {e}")

