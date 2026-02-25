# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import numpy as np
import time
import torch
import torch.distributed as dist
import subprocess
import imageio
import librosa
import numpy as np
from loguru import logger
from collections import deque
from datetime import datetime

from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify FlashHead model checkpoint directory."
    assert args.wav2vec_dir is not None, "Please specify the wav2vec checkpoint directory."
    assert args.model_type=="pro" or args.model_type=="lite", "Please specify the model name (pro, lite)."
    assert args.cond_image_dir is not None or args.cond_image is not None, "Please specify the condition image or directory."
    assert args.audio_path is not None, "Please specify the audio path."

    args.base_seed = args.base_seed if args.base_seed >= 0 else 42

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video from one image using FlashHead"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to FlashHead model checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default=None,
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Choose from pro or lite.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--cond_image",
        type=str,
        default=None,
        help="[meta file] The condition image path to generate the video.")
    parser.add_argument(
        "--cond_image_dir",
        type=str,
        default=None,
        help="[meta directory] The directory of condition images.")
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="[meta file] The audio path to generate the video.")
    parser.add_argument(
        "--audio_encode_mode",
        type=str,
        default="stream",
        choices=['stream', 'once'],
        help="stream: encode audio chunk before every generation; once: encode audio together")
    parser.add_argument(
        "--use_face_crop",
        type=bool,
        default=False,
        help="Enable face detection and crop for condition image")
    args = parser.parse_args()
    _validate_args(args)

    return args

def save_video(frames_list, video_path, audio_path, fps):
    temp_video_path = video_path.replace('.mp4', '_tmp.mp4')
    with imageio.get_writer(temp_video_path, format='mp4', mode='I',
                            fps=fps , codec='h264', ffmpeg_params=['-bf', '0']) as writer:
        for frames in frames_list:
            frames = frames.numpy().astype(np.uint8)
            for i in range(frames.shape[0]):
                frame = frames[i, :, :, :]
                writer.append_data(frame)
    
    # merge video and audio
    cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', video_path, '-y']
    subprocess.run(cmd)
    os.remove(temp_video_path)


def generate(args):
    run_start = time.perf_counter()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    load_start = time.perf_counter()
    pipeline = get_pipeline(world_size=world_size, ckpt_dir=args.ckpt_dir, wav2vec_dir=args.wav2vec_dir, model_type=args.model_type)
    get_base_data(pipeline, cond_image_path_or_dir=args.cond_image_dir if args.cond_image_dir is not None else args.cond_image, base_seed=args.base_seed, use_face_crop=args.use_face_crop)
    load_time = time.perf_counter() - load_start
    if rank == 0:
        logger.info(f"Model load + data prep: {load_time:.2f}s")

    infer_params = get_infer_params()
    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num

    t0 = time.perf_counter()
    human_speech_array_all, _ = librosa.load(args.audio_path, sr=infer_params['sample_rate'], mono=True)
    audio_load_time = time.perf_counter() - t0
    if rank == 0:
        logger.info(f"Audio load: {audio_load_time:.2f}s")

    total_inference_time = 0.0
    if rank == 0:
        logger.info("Data preparation done. Start to generate video...")
        logger.info("First chunk may be slow (JIT/torch.compile warmup); later chunks reflect steady-state speed.")

    generated_list = []
    if args.audio_encode_mode == 'once':
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        audio_embedding_all = get_audio_embedding(pipeline, human_speech_array_all)
        torch.cuda.synchronize()
        audio_encode_time = time.perf_counter() - start_time
        total_inference_time += audio_encode_time
        if rank == 0:
            logger.info(f"Audio encode (once): {audio_encode_time:.2f}s")

        audio_embedding_chunks_list = [audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous() for i in range((audio_embedding_all.shape[1]-frame_num) // slice_len)]
        for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            video = run_pipeline(pipeline, audio_embedding_chunk)
            torch.cuda.synchronize()
            chunk_time = time.perf_counter() - start_time
            total_inference_time += chunk_time
            if rank == 0:
                logger.info(f"Generate video chunk-{chunk_idx} done, cost time: {chunk_time:.2f}s")
            generated_list.append(video.cpu())

    elif args.audio_encode_mode == 'stream':
        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num

        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
        human_speech_array_slices = human_speech_array_all[:(len(human_speech_array_all)//(human_speech_array_slice_len))*human_speech_array_slice_len].reshape(-1, human_speech_array_slice_len)

        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
            video = run_pipeline(pipeline, audio_embedding)

            torch.cuda.synchronize()
            chunk_time = time.perf_counter() - start_time
            total_inference_time += chunk_time
            if rank == 0:
                logger.info(f"Generate video chunk-{chunk_idx} done (audio encode + inference): {chunk_time:.2f}s")
            generated_list.append(video.cpu())

    if rank == 0:
        logger.info(f"Total inference time (excl. load, incl. audio encode): {total_inference_time:.2f}s")
        if args.save_file is None:
            output_dir = 'sample_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S-%f")[:-3]
            filename = f"res_{timestamp}.mp4"
            filepath = os.path.join(output_dir, filename)
            args.save_file = filepath

        start_save = time.perf_counter()
        save_video(generated_list, args.save_file, args.audio_path, fps=tgt_fps)
        save_time = time.perf_counter() - start_save
        total_wall = time.perf_counter() - run_start

        logger.info(f"Save time (encode + ffmpeg): {save_time:.2f}s")
        logger.info(f"Saving generated video to {args.save_file}")
        logger.info(
            "Run complete | load: {:.2f}s | audio_load: {:.2f}s | inference: {:.2f}s | save: {:.2f}s | total (wall): {:.2f}s".format(
                load_time, audio_load_time, total_inference_time, save_time, total_wall
            )
        )
        logger.info("Finished.")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    args = _parse_args()
    generate(args)