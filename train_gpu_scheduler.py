

import subprocess
import time
import os
import random

def get_gpu_memory_usage(gpu_idx):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_idx)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return int(result.stdout.strip())
    except Exception as e:
        print(f"Error checking GPU {gpu_idx}: {e}")
        return None

def wait_for_training_completion(gpu_pair, threshold=10000):
    """
    지정된 GPU 쌍의 메모리 사용량이 특정 임계값(threshold) 이하로 떨어질 때까지 대기
    """
    print(f"Monitoring GPUs {gpu_pair} for completion...")
    while True:
        mem_usages = [get_gpu_memory_usage(gpu) for gpu in gpu_pair]
        if all(mem is not None and mem < threshold for mem in mem_usages):
            print(f"GPUs {gpu_pair} are now available (Memory usage: {mem_usages}).")
            return
        time.sleep(30)  # 30초마다 확인

def start_training(gpu_pair, config_path):
    """
    지정된 두 개의 GPU에서 학습 스크립트를 실행하면서 MASTER_ADDR과 PORT를 동적으로 할당
    """
    print(f"Starting training on GPUs {gpu_pair} with config {config_path}...")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_pair))
    
    # 동적 포트 및 마스터 주소 설정
    master_addr = f"127.0.0.{random.randint(2, 254)}"
    master_port = str(random.randint(10000, 60000))

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    command = ["tools/dist_train_gpu_scheduler.sh", config_path, "2"]
    subprocess.run(command)

def main(gpu_pair, config_paths):
    for config in config_paths:
        wait_for_training_completion(gpu_pair)  # 이전 학습이 끝날 때까지 기다림
        start_training(gpu_pair, config)        # 학습 시작
        print(f"Training on {config} completed.")

if __name__ == "__main__":
    GPU_PAIR = [2, 3]  # 사용할 GPU 쌍
    CONFIG_PATHS = [
        "configs/jy/objectness.py",
        "configs/jy/objectness-ver2.py",
        "configs/jy/objectness-ver2-w10.py",
        "configs/jy/objectness-ver2-sgd.py"
    ]  # 학습할 config 파일 리스트
    
    main(GPU_PAIR, CONFIG_PATHS)
