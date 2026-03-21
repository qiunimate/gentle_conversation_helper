import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
import numpy as np
import os

# 1. 强制使用 16000Hz (Whisper 标准)
TARGET_RATE = 16000
CHUNK_SIZE = 1024 * 4  # 每次读取的块大小

model = WhisperModel("small", device="cuda", compute_type="float16")

def start_listening():
    p = pyaudio.PyAudio()
    
    # 找到正确的 Loopback 设备
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    
    loopback_device = None
    for loopback in p.get_loopback_device_info_generator():
        if default_speakers["name"] in loopback["name"]:
            loopback_device = loopback
            break

    # 获取设备原生的采样率（通常是 48000）和声道数（通常是 2）
    device_rate = int(loopback_device["defaultSampleRate"])
    channels = loopback_device["maxInputChannels"]

    print(f"正在监听设备: {loopback_device['name']}")
    print(f"原始采样率: {device_rate}, 声道数: {channels}")

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=device_rate,  # 必须用设备原生的采样率打开
                    input=True,
                    input_device_index=loopback_device["index"])

    audio_buffer = []

    while True:
        # 读取约 3 秒的数据量
        # 数据量 = 采样率 * 声道数 * 秒数
        raw_data = stream.read(device_rate * 3)
        
        # 将字节转为 Int16，再归一化到 Float32
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 【关键步骤 1】如果是多声道，合并为单声道
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        
        # 【关键步骤 2】如果采样率不是 16000，必须进行重采样
        # 简单粗暴的降采样（如果 device_rate 是 48000，即每 3 个点取 1 个）
        if device_rate != TARGET_RATE:
            samples = samples[::int(device_rate / TARGET_RATE)]

        # 再次检查音量峰值
        peak = np.max(np.abs(samples))
        if peak < 0.01: # 过滤太小的声音
            continue

        # 3. 运行识别
        segments, _ = model.transcribe(samples, beam_size=5, vad_filter=True, language="en")
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

if __name__ == "__main__":
    start_listening()