"""Lazy, explicitly placed Whisper transcriber with deterministic release."""
import concurrent.futures
import gc
import math
import os
import re
from typing import List, Dict, Any, Optional

from search.device_detector import configured_devices, release_cuda


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        window_sec: float = 10.0,
        languages: Optional[List[str]] = None,
        device: Optional[str] = None,
        autoload: bool = False,
    ):
        self.model_size = model_size
        self.window_sec = float(window_sec)
        self.languages = languages or ["ar-SA", "en-US"]
        self.device = device or configured_devices()["ingest"]
        self.model = None
        self.backend = "none"
        if autoload:
            self.load_model()

    def load_model(self):
        if self.model is not None:
            return self
        if self.device.startswith("cuda"):
            device_type = "cuda"
            device_index = int(self.device.split(":")[-1])
            compute_type = "float16"
        else:
            device_type = "cpu"
            device_index = 0
            compute_type = "int8"

        try:
            from faster_whisper import WhisperModel
            print(
                f"[AudioTranscriber] Loading Whisper {self.model_size} "
                f"on {self.device} ({compute_type})..."
            )
            self.model = WhisperModel(
                self.model_size,
                device=device_type,
                device_index=device_index,
                compute_type=compute_type,
            )
            self.backend = "faster_whisper"
            print(f"[AudioTranscriber]  Loaded on {self.device}")
            return self
        except Exception as exc:
            print(f"[AudioTranscriber] Large model unavailable: {exc}")

        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel("base", device="cpu", compute_type="int8")
            self.backend = "faster_whisper_cpu"
            self.device = "cpu"
            print("[AudioTranscriber]  CPU base fallback loaded")
        except Exception as exc:
            print(f"[AudioTranscriber] No local Whisper model: {exc}")
            self.model = None
        return self

    def release_model(self):
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
        self.model = None
        self.backend = "released"
        gc.collect()
        release_cuda(self.device)
        print(f"[AudioTranscriber]  Released model from {self.device}")

    def _recognize_chunk(self, audio_path, start_sec, chunk_len):
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        output = []
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source, offset=start_sec, duration=chunk_len)
            text = ""
            for language in self.languages:
                try:
                    text = recognizer.recognize_google(audio_data, language=language)
                    if text:
                        break
                except Exception:
                    continue
            words = text.split()
            duration = chunk_len / max(len(words), 1)
            for index, word in enumerate(words):
                start = start_sec + index * duration
                output.append({
                    "word": word.strip(),
                    "start": round(start, 3),
                    "end": round(start + duration, 3),
                })
        except Exception:
            pass
        return output

    def transcribe_audio(self, audio_path):
        if not os.path.exists(audio_path):
            return []
        self.load_model()
        words = []

        if self.model is not None:
            try:
                segments, _ = self.model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    vad_filter=True,
                )
                for segment in segments:
                    for word in segment.words or []:
                        words.append({
                            "word": word.word.strip(),
                            "start": round(float(word.start), 3),
                            "end": round(float(word.end), 3),
                        })
                if words:
                    return words
            except Exception as exc:
                print(f"[AudioTranscriber] Local transcription fallback: {exc}")

        try:
            import wave
            with wave.open(audio_path, "rb") as handle:
                duration = handle.getnframes() / float(handle.getframerate())
            chunks = []
            for index in range(math.ceil(duration / self.window_sec)):
                start = index * self.window_sec
                length = min(self.window_sec, duration - start)
                if length > 0.5:
                    chunks.append((start, length))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self._recognize_chunk, audio_path, start, length)
                    for start, length in chunks
                ]
                for future in concurrent.futures.as_completed(futures):
                    words.extend(future.result())
            return sorted(words, key=lambda item: item["start"])
        except Exception as exc:
            print(f"[AudioTranscriber] ASR fallback failed: {exc}")
            return []

    @staticmethod
    def detect_sentence_boundaries(word_timestamps):
        boundaries = []
        punctuation = re.compile(r"[.?!;؛؟]")
        for index, word in enumerate(word_timestamps):
            if punctuation.search(word["word"]):
                boundaries.append(word["end"])
            elif index + 1 < len(word_timestamps):
                if word_timestamps[index + 1]["start"] - word["end"] > 1.5:
                    boundaries.append(word["end"])
        return sorted(set(boundaries))

    @staticmethod
    def align_transcripts_to_chunks(clips, word_timestamps):
        for clip in clips:
            start = float(clip.get("start_ts", 0.0))
            end = float(clip.get("end_ts", 0.0))
            selected = [
                word for word in word_timestamps
                if start <= float(word["start"]) < end
                or start < float(word["end"]) <= end
            ]
            clip["word_timestamps"] = selected
            clip["transcript"] = " ".join(item["word"] for item in selected).strip()
        return clips
