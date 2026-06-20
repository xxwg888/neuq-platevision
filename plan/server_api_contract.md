# Server API Contract

## Remote Inference Endpoint

本地后端会将 `remote_server` provider 请求转发到设置中的 `endpoint`。

### Request

`POST {endpoint}`

FormData:

- `file`: 图片文件
- `provider`: 固定为 `remote_server`
- `return_intermediate`: `true` 或 `false`

### Response

返回 JSON 结构需要兼容本地 `POST /api/recognize`：

```json
{
  "request_id": "server_001",
  "provider": "remote_server",
  "provider_used": "remote_server",
  "plate_text": "冀A12345",
  "plate_type": "blue",
  "confidence": 0.94,
  "bbox": [120, 230, 420, 80],
  "chars": [
    {"text": "冀", "confidence": 0.91, "bbox": [0, 0, 20, 40]}
  ],
  "images": {
    "detected": null,
    "plate_crop": null,
    "mask": null,
    "binary": null,
    "segmented": null
  },
  "timing_ms": {
    "detect": 35.0,
    "recognize": 42.0,
    "total": 77.0
  },
  "messages": []
}
```

## Notes

- 服务器可以先只返回 `plate_text`、`confidence`、`bbox`，其余字段按空值补齐。
- 如果服务器返回图片 URL，前端可以直接显示绝对 URL。
- 如果远程调用失败，本地后端会自动回退到 `opencv_baseline`。

