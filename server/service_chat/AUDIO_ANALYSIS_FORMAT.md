# Định Dạng Dữ Liệu Audio Analysis

## Tổng Quan

Sau khi upload file audio và service phân tích xong, server sẽ trả về response với format JSON. Dữ liệu audio analysis được gửi kèm trong field `audio_analysis` của response.

## Response Format từ `/upload` Endpoint

### Khi upload thành công (Status 200)

```json
{
  "file_id": "uuid-filename.wav",
  "filename": "example.wav",
  "file_type": "audio",
  "file_size": 3840092,
  "uploaded_at": "2025-12-26T01:30:00.123456",
  "url": "/files/uuid-filename.wav",
  "audio_analysis": {
    "metadata": {
      "duration": 120.5,
      "sample_rate": 44100,
      "channels": 2
    },
    "acoustics_information": [
      {
        "start_time": 0.0,
        "end_time": 30.0,
        "background_scene": "indoor, office",
        "sound_events": [
          {"event": "door_slam", "start": 5.2, "end": 5.5},
          {"event": "footsteps", "start": 10.0, "end": 15.0}
        ]
      },
      {
        "start_time": 30.0,
        "end_time": 60.0,
        "background_scene": "outdoor, street",
        "sound_events": [
          {"event": "car_horn", "start": 35.0, "end": 35.2}
        ]
      }
    ],
    "human_speech_information": {
      "human speech information": [
        {
          "text": "Xin chào, tôi là người dùng",
          "start_time": 2.5,
          "end_time": 5.0,
          "speaker": "speaker_0",
          "emotion": "neutral",
          "gender": "male"
        },
        {
          "text": "Bạn có thể giúp tôi không?",
          "start_time": 6.0,
          "end_time": 9.5,
          "speaker": "speaker_0",
          "emotion": "happy",
          "gender": "male"
        }
      ],
      "language detection": "vi",
      "number of speakers": 1
    },
    "other_information": {
      "deepfake detection": "real"
    }
  }
}
```

### Khi không có audio analysis (file không phải audio hoặc services không chạy)

```json
{
  "file_id": "uuid-document.pdf",
  "filename": "document.pdf",
  "file_type": "document",
  "file_size": 102400,
  "uploaded_at": "2025-12-26T01:30:00.123456",
  "url": "/files/uuid-document.pdf",
  "audio_analysis": null
}
```

## Cấu Trúc Chi Tiết của `audio_analysis`

### 1. `metadata` (Object hoặc null)

Thông tin cơ bản về file audio:

```json
{
  "duration": 120.5,        // Thời lượng (giây)
  "sample_rate": 44100,     // Tần số lấy mẫu (Hz)
  "channels": 2             // Số kênh (1 = mono, 2 = stereo)
}
```

**Nguồn:** Whisper service

### 2. `acoustics_information` (Array hoặc null)

Thông tin về âm thanh nền và sự kiện âm thanh, được chia theo từng đoạn thời gian:

```json
[
  {
    "start_time": 0.0,      // Thời điểm bắt đầu (giây)
    "end_time": 30.0,       // Thời điểm kết thúc (giây)
    "background_scene": "indoor, office",  // Cảnh nền
    "sound_events": [       // Danh sách sự kiện âm thanh
      {
        "event": "door_slam",
        "start": 5.2,
        "end": 5.5
      },
      {
        "event": "footsteps",
        "start": 10.0,
        "end": 15.0
      }
    ]
  }
]
```

**Nguồn:** 
- `background_scene` và `sound_events`: ASC-AED service
- Cấu trúc timeline: CAP-DF service (audio_captioning)

### 3. `human_speech_information` (Object hoặc null)

Thông tin về giọng nói con người:

```json
{
  "human speech information": [
    {
      "text": "Nội dung được nhận dạng",
      "start_time": 2.5,    // Thời điểm bắt đầu (giây)
      "end_time": 5.0,      // Thời điểm kết thúc (giây)
      "speaker": "speaker_0",  // ID người nói
      "emotion": "neutral",    // Cảm xúc: neutral, happy, sad, angry, etc.
      "gender": "male"         // Giới tính: male, female
    }
  ],
  "language detection": "vi",  // Ngôn ngữ được phát hiện
  "number of speakers": 1      // Số lượng người nói
}
```

**Nguồn:** Whisper service

**Lưu ý:** 
- `language detection` có thể là string (ví dụ: "vi", "en") hoặc null
- `number of speakers` có thể là số hoặc null

### 4. `other_information` (Object hoặc null)

Thông tin khác:

```json
{
  "deepfake detection": "real"  // "real", "fake", hoặc null
}
```

**Nguồn:** CAP-DF service (deepfake_detection)

**Giá trị có thể:**
- `"real"`: Audio là thật
- `"fake"`: Audio là giả/deepfake
- `null`: Không xác định được

## Cách Flutter App Nhận Dữ Liệu

### 1. Trong `FileUploadService.uploadFile()`

```dart
final jsonResponse = jsonDecode(response.body) as Map<String, dynamic>;

final attachment = FileAttachment(
  fileId: jsonResponse['file_id'] as String,
  filename: jsonResponse['filename'] as String,
  fileType: jsonResponse['file_type'] as String,
  fileSize: jsonResponse['file_size'] as int,
  url: jsonResponse['url'] as String,
  uploadedAt: DateTime.parse(jsonResponse['uploaded_at'] as String),
  audioAnalysis: jsonResponse['audio_analysis'] as Map<String, dynamic>?,  // ← Đây
);
```

### 2. Trong `FileAttachment` Model

```dart
class FileAttachment {
  final Map<String, dynamic>? audioAnalysis;  // Lưu trữ toàn bộ audio_analysis object
  
  // ...
}
```

### 3. Sử dụng trong Chat

Khi gửi tin nhắn có file audio đính kèm, app sẽ:

1. Lấy `audioAnalysis` từ `FileAttachment`
2. Truyền vào `ChatApiService.streamChat()` qua parameter `audioAnalysisData`
3. Server sẽ sử dụng dữ liệu này để tạo context cho LLM

```dart
// Trong chat_detail_screen.dart
Map<String, dynamic>? audioAnalysisData;

for (final att in attachmentsToSend) {
  if (att.fileType == 'audio' && att.audioAnalysis != null) {
    audioAnalysisData = att.audioAnalysis;  // ← Lấy audio analysis
    break;
  }
}

// Gửi kèm khi chat
final fullResponse = await ChatApiService.streamChat(
  messages: apiMessages,
  audioAnalysisData: audioAnalysisData,  // ← Truyền vào API
  onChunk: (chunk) { ... },
);
```

## Xử Lý Trường Hợp Null/Missing

### Khi `audio_analysis` là `null`:

- File không phải audio (image, document, etc.)
- Audio processing services không chạy
- Lỗi xảy ra trong quá trình xử lý

**App sẽ:**
- Vẫn upload file thành công
- `audioAnalysis` trong `FileAttachment` sẽ là `null`
- Chat vẫn hoạt động bình thường, chỉ không có context từ audio analysis

### Khi một phần dữ liệu là `null`:

- `metadata` có thể null nếu Whisper service lỗi
- `acoustics_information` có thể null nếu ASC-AED hoặc CAP-DF lỗi
- `human_speech_information` có thể null nếu Whisper service lỗi
- `other_information` có thể null nếu CAP-DF lỗi

**Server sẽ:**
- Vẫn trả về `audio_analysis` object
- Các field bị lỗi sẽ là `null` hoặc empty array/object
- Không throw exception, chỉ log warning

## Ví Dụ Thực Tế

### Response khi upload file audio thành công:

```json
{
  "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890.wav",
  "filename": "recording.wav",
  "file_type": "audio",
  "file_size": 3840092,
  "uploaded_at": "2025-12-26T01:30:00.123456",
  "url": "/files/a1b2c3d4-e5f6-7890-abcd-ef1234567890.wav",
  "audio_analysis": {
    "metadata": {
      "duration": 87.5,
      "sample_rate": 44100,
      "channels": 1
    },
    "acoustics_information": [
      {
        "start_time": 0.0,
        "end_time": 87.5,
        "background_scene": "indoor, quiet",
        "sound_events": []
      }
    ],
    "human_speech_information": {
      "human speech information": [
        {
          "text": "Xin chào",
          "start_time": 1.0,
          "end_time": 2.5,
          "speaker": "speaker_0",
          "emotion": "neutral",
          "gender": "male"
        }
      ],
      "language detection": "vi",
      "number of speakers": 1
    },
    "other_information": {
      "deepfake detection": "real"
    }
  }
}
```

### Response khi services không chạy:

```json
{
  "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890.wav",
  "filename": "recording.wav",
  "file_type": "audio",
  "file_size": 3840092,
  "uploaded_at": "2025-12-26T01:30:00.123456",
  "url": "/files/a1b2c3d4-e5f6-7890-abcd-ef1234567890.wav",
  "audio_analysis": null
}
```

## Tóm Tắt

1. **Format:** JSON object trong field `audio_analysis` của upload response
2. **Cấu trúc:** 4 phần chính:
   - `metadata`: Thông tin file audio
   - `acoustics_information`: Âm thanh nền và sự kiện
   - `human_speech_information`: Giọng nói và transcript
   - `other_information`: Deepfake detection
3. **Null handling:** Tất cả fields đều có thể null
4. **App usage:** Lưu trong `FileAttachment.audioAnalysis` và truyền vào chat API

