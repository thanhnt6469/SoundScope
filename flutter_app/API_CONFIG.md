# Cấu hình API URL

## Vấn đề
Khi chạy Flutter app trên Android device thật, app không thể kết nối tới server nếu dùng `localhost` vì `localhost` sẽ trỏ tới chính device, không phải máy tính chạy server.

## Giải pháp

### Bước 1: Tìm IP của máy tính chạy server

**Windows:**
```powershell
ipconfig
```
Tìm dòng `IPv4 Address` trong phần adapter đang kết nối (thường là WiFi hoặc Ethernet). Ví dụ: `192.168.1.100`

**Linux/Mac:**
```bash
ifconfig
# hoặc
ip addr
```
Tìm IP trong phần adapter đang kết nối.

### Bước 2: Cập nhật IP trong code

Mở file `lib/constants/api_constants.dart` và thay đổi:

```dart
static const String baseUrlDev = 'http://YOUR_IP_HERE:8003';
```

Ví dụ nếu IP là `192.168.1.100`:
```dart
static const String baseUrlDev = 'http://192.168.1.100:8003';
```

### Bước 3: Đảm bảo server đang chạy

Chạy server chat API:
```bash
cd server/service_chat
py -m uvicorn chat_api:app --host 0.0.0.0 --port 8003
```

**Quan trọng:** Phải dùng `--host 0.0.0.0` để server lắng nghe trên tất cả interfaces, không chỉ localhost.

### Bước 4: Kiểm tra firewall

Đảm bảo firewall cho phép kết nối tới port 8003:
- Windows: Mở Windows Defender Firewall và cho phép port 8003
- Linux: `sudo ufw allow 8003`

### Bước 5: Kiểm tra kết nối

Từ device Android, mở browser và truy cập:
```
http://YOUR_IP:8003
```

Nếu thấy JSON response `{"message":"Chat API Service - Streaming LLM Chat"}`, server đã sẵn sàng.

## Lưu ý

- Device Android và máy tính phải cùng mạng WiFi
- IP có thể thay đổi khi kết nối lại WiFi, cần cập nhật lại nếu cần
- Trong production, nên dùng domain name thay vì IP

