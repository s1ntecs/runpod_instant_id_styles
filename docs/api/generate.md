## Request

```json
{
  "input": {
    "model": "wangqixun/YamerMIX_v8",
    "face_image": "base64 encoded image content",
    "pose_image": "base64 encoded image content",
    "prompt": "A man",
    "negative_prompt": "nsfw",
    "style_name": "Watercolor",
    "num_steps": 30,
    "identitynet_strength_ratio": 0.8,
    "adapter_strength_ratio":  0.8,
    "guidance_scale": 5,
    "seed": 42
  }
}
```

## Response

## RUN

```json
{
  "id": "83bbc301-5dcd-4236-9293-a65cdd681858",
  "status": "IN_QUEUE"
}
```

## RUNSYNC

```json
{
  "delayTime": 7334,
  "executionTime": 20260,
  "id": "sync-acccda60-3017-41c8-bf93-d495ab1c0557-e1",
  "output": {
    "image": "base64 encoded result image"
  },
  "status": "COMPLETED"
}
```
