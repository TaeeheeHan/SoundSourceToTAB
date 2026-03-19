"""
음정 감지 모듈
Primary: Spotify Basic Pitch (ONNX 백엔드, TF 불필요)
Fallback: librosa pyin (개선된 파라미터)
"""
from __future__ import annotations

import os
import warnings
import pathlib
import numpy as np
import librosa
from scipy.signal import medfilt, butter, sosfilt
from dataclasses import dataclass
from typing import List

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# ── 베이스 범위 ──────────────────────────────────────────────────────
BASS_MIDI_MIN = 28   # E1  (41.2 Hz)
BASS_MIDI_MAX = 55   # G3  (196 Hz)  — 표준 4현 베이스 최고음
BASS_HZ_MIN   = 35.0
BASS_HZ_MAX   = 220.0


# ────────────────────────────────────────────────────────────────────
@dataclass
class NoteEvent:
    pitch_midi: int
    start_time: float   # 초
    end_time:   float   # 초
    velocity:   float = 0.8
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ────────────────────────────────────────────────────────────────────
def _get_onnx_model_path() -> str | None:
    """basic-pitch 패키지 내부의 ONNX 모델 경로를 반환."""
    try:
        import basic_pitch
        p = pathlib.Path(basic_pitch.__file__).parent / "saved_models/icassp_2022/nmp.onnx"
        return str(p) if p.exists() else None
    except ImportError:
        return None


def _basic_pitch_available() -> bool:
    try:
        import basic_pitch          # noqa: F401
        import onnxruntime          # noqa: F401
        return _get_onnx_model_path() is not None
    except ImportError:
        return False


def _crepe_available() -> bool:
    try:
        import crepe                # noqa: F401
        return True
    except ImportError:
        return False


# ────────────────────────────────────────────────────────────────────
class PitchDetector:
    def __init__(self, method: str = 'auto'):
        """
        method: 'auto' | 'basic-pitch' | 'librosa' | 'crepe'
        auto → crepe → basic-pitch → librosa 순으로 시도
        """
        self.method = method

    # ── 공개 API ─────────────────────────────────────────────────────
    def detect_notes(self, audio_path: str,
                     min_confidence: float = 0.25) -> List[NoteEvent]:
        m = self.method

        # CREPE 강제 지정
        if m == 'crepe':
            notes = self._detect_crepe(audio_path)
            return [n for n in notes if n.confidence >= min_confidence]

        # Basic-Pitch 강제 지정
        if m == 'basic-pitch':
            try:
                notes = self._detect_basic_pitch(audio_path)
                return [n for n in notes if n.confidence >= min_confidence]
            except Exception as exc:
                print(f"[basic-pitch] 실패 ({exc}), librosa로 전환")
            notes = self._detect_librosa(audio_path)
            return [n for n in notes if n.confidence >= min_confidence]

        # auto: crepe → basic-pitch → librosa
        if m == 'auto':
            if _crepe_available():
                try:
                    notes = self._detect_crepe(audio_path)
                    return [n for n in notes if n.confidence >= min_confidence]
                except Exception as exc:
                    print(f"[crepe] 실패 ({exc}), 다음 방법으로 전환")
            if _basic_pitch_available():
                try:
                    notes = self._detect_basic_pitch(audio_path)
                    return [n for n in notes if n.confidence >= min_confidence]
                except Exception as exc:
                    print(f"[basic-pitch] 실패 ({exc}), librosa로 전환")

        # librosa fallback
        notes = self._detect_librosa(audio_path)
        return [n for n in notes if n.confidence >= min_confidence]

    # ── CREPE (딥러닝) 백엔드 ────────────────────────────────────────
    def _detect_crepe(self, audio_path: str) -> List[NoteEvent]:
        """
        CREPE: 딥러닝 기반 단음 F0 추정기 (Kereliuk et al., 2018).
        librosa pyin보다 저음역 정확도가 월등히 높음.
        pip install crepe 필요.
        """
        import crepe

        SR = 16000   # CREPE 권장 샘플링 레이트
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        y = librosa.util.normalize(y)

        # 베이스 대역 밴드패스 필터
        y_filt = self._bandpass(y, SR, 28, 380)

        # CREPE 추론 (viterbi=True → 옥타브 점프 억제)
        time_arr, freq_arr, conf_arr, _ = crepe.predict(
            y_filt, SR,
            viterbi=True,
            step_size=10,   # 10ms 단위 프레임
            verbose=0,
        )

        # 베이스 범위 + 신뢰도 필터로 voiced 마스크 생성
        in_range    = (freq_arr >= BASS_HZ_MIN) & (freq_arr <= BASS_HZ_MAX)
        voiced_flag = (conf_arr > 0.45) & in_range
        f0          = np.where(voiced_flag, freq_arr, np.nan)

        # 온셋 검출 (note segmentation 기준)
        HOP = 256
        y_orig, sr_orig = librosa.load(audio_path, sr=22050, mono=True)
        onset_frames = self._detect_onsets(y_orig, sr_orig, HOP)
        onset_times  = librosa.frames_to_time(onset_frames, sr=sr_orig,
                                               hop_length=HOP)

        notes = self._build_notes(f0, voiced_flag, conf_arr,
                                   time_arr, onset_times)
        notes = self._filter_short(notes, min_dur=0.06)
        notes = self._merge_same_pitch(notes, max_gap=0.10)
        return notes

    # ── Basic-Pitch (ONNX) 백엔드 ────────────────────────────────────
    def _detect_basic_pitch(self, audio_path: str) -> List[NoteEvent]:
        from basic_pitch.inference import predict

        onnx_path = _get_onnx_model_path()
        _, midi_data, _ = predict(
            audio_path,
            onnx_path,
            minimum_frequency=BASS_HZ_MIN,
            maximum_frequency=BASS_HZ_MAX,
            minimum_note_length=60,     # ms
            onset_threshold=0.5,
            frame_threshold=0.3,
        )

        notes: List[NoteEvent] = []
        for inst in midi_data.instruments:
            for n in inst.notes:
                if BASS_MIDI_MIN <= n.pitch <= BASS_MIDI_MAX:
                    notes.append(NoteEvent(
                        pitch_midi=n.pitch,
                        start_time=float(n.start),
                        end_time=float(n.end),
                        velocity=n.velocity / 127.0,
                        confidence=1.0,
                    ))

        notes.sort(key=lambda n: n.start_time)
        notes = self._remove_octave_duplicates(notes)
        notes = self._merge_same_pitch(notes, max_gap=0.08)
        notes = self._filter_short(notes, min_dur=0.06)
        return notes

    # ── librosa (pyin) 백엔드 ────────────────────────────────────────
    def _detect_librosa(self, audio_path: str) -> List[NoteEvent]:
        SR    = 22050
        HOP   = 256
        FRAME = 8192    # 저주파(35Hz) 해상도 향상 — 4096보다 안정적

        y, sr = librosa.load(audio_path, sr=SR, mono=True)
        y = librosa.util.normalize(y)

        # 베이스 주파수 대역 필터
        y_filt = self._bandpass(y, SR, 28, 380)

        # 하모닉 분리 (배음 혼동 방지) — margin 낮춰 지나친 제거 방지
        y_harm = librosa.effects.harmonic(y_filt, margin=2.0)

        # 어택 검출 — 베이스 대역 신호로, delta 낮춰 더 많은 온셋 포착
        onset_frames = self._detect_onsets(y_filt, SR, HOP)
        onset_times  = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP)

        # pyin 피치 추적
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_harm,
            fmin=BASS_HZ_MIN,
            fmax=BASS_HZ_MAX,
            sr=SR,
            hop_length=HOP,
            frame_length=FRAME,
            fill_na=None,
        )
        times = librosa.times_like(f0, sr=SR, hop_length=HOP)

        # voiced_prob 기반 완화된 마스크 (pyin voiced_flag는 너무 보수적)
        is_voiced = (voiced_prob >= 0.10) & (~np.isnan(f0))

        # f0 스무딩
        f0_smooth = self._smooth_f0(f0, is_voiced)

        # 노트 이벤트 생성
        notes = self._build_notes(f0_smooth, is_voiced, voiced_prob,
                                   times, onset_times)
        notes = self._filter_short(notes, min_dur=0.04)
        notes = self._merge_same_pitch(notes, max_gap=0.08)
        return notes

    # ── 전처리 유틸 ──────────────────────────────────────────────────
    def _bandpass(self, y: np.ndarray, sr: int, flo: float, fhi: float) -> np.ndarray:
        try:
            nyq = sr / 2.0
            sos = butter(4, [flo / nyq, fhi / nyq], btype='bandpass', output='sos')
            return sosfilt(sos, y).astype(y.dtype)
        except Exception:
            return y

    def _detect_onsets(self, y: np.ndarray, sr: int, hop: int) -> np.ndarray:
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop,
            fmax=350, aggregate=np.median,
        )
        return librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr, hop_length=hop,
            backtrack=True, units='frames',
            delta=0.07, wait=1,   # 낮은 delta → 약한 베이스 어택도 포착
        )

    def _smooth_f0(self, f0: np.ndarray, voiced_flag: np.ndarray) -> np.ndarray:
        out  = np.copy(f0)
        mask = voiced_flag & (~np.isnan(f0))
        if np.sum(mask) < 3:
            return out
        idx_v = np.where(mask)[0]
        f0_interp = np.interp(np.arange(len(f0)), idx_v, f0[mask])
        smoothed  = medfilt(f0_interp, kernel_size=5)
        return np.where(mask, smoothed, np.nan)

    # ── 노트 이벤트 생성 ─────────────────────────────────────────────
    def _build_notes(self, f0, voiced_flag, voiced_prob,
                     times, onset_times) -> List[NoteEvent]:
        notes: List[NoteEvent] = []
        if len(onset_times) == 0:
            return notes

        total_t = float(times[-1]) if len(times) > 0 else 0.0

        for i, t0 in enumerate(onset_times):
            t1        = float(onset_times[i + 1]) if i + 1 < len(onset_times) else total_t
            time_mask = (times >= t0) & (times < t1)
            mask      = time_mask & voiced_flag

            # voiced_flag가 너무 엄격해 구간 전체 미검출 시 → prob 기반 폴백
            if not np.any(mask):
                mask = time_mask & (voiced_prob >= 0.10) & (~np.isnan(f0))

            if not np.any(mask):
                continue

            f0_seg   = f0[mask]
            prob_seg = voiced_prob[mask]
            valid    = ~np.isnan(f0_seg)

            if not np.any(valid):
                continue

            median_f0 = float(np.median(f0_seg[valid]))
            conf      = float(np.mean(prob_seg[valid]))
            midi      = int(round(librosa.hz_to_midi(median_f0)))

            if BASS_MIDI_MIN <= midi <= BASS_MIDI_MAX:
                notes.append(NoteEvent(
                    pitch_midi=midi,
                    start_time=float(t0),
                    end_time=float(t1),
                    confidence=conf,
                ))
        return notes

    # ── 후처리 ───────────────────────────────────────────────────────
    def _remove_octave_duplicates(self, notes: List[NoteEvent],
                                  time_window: float = 0.05) -> List[NoteEvent]:
        """
        동시에 발생한 옥타브 중복 음표 제거 (낮은 음 우선 보존).
        basic-pitch가 기음 + 배음을 동시에 감지하는 경우를 처리.
        """
        if not notes:
            return notes

        keep = [True] * len(notes)
        for i in range(len(notes)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(notes)):
                if notes[j].start_time - notes[i].start_time > time_window:
                    break
                diff = abs(notes[j].pitch_midi - notes[i].pitch_midi)
                # 옥타브 또는 5도 배음 관계
                if diff in (12, 19, 24):
                    # 높은 음 제거 (낮은 음이 기음)
                    if notes[i].pitch_midi < notes[j].pitch_midi:
                        keep[j] = False
                    else:
                        keep[i] = False

        return [n for n, k in zip(notes, keep) if k]

    def _merge_same_pitch(self, notes: List[NoteEvent],
                          max_gap: float = 0.10) -> List[NoteEvent]:
        """같은 음정의 연속 음표를 하나로 합침."""
        if len(notes) <= 1:
            return notes
        result = [notes[0]]
        for note in notes[1:]:
            last = result[-1]
            gap  = note.start_time - last.end_time
            if last.pitch_midi == note.pitch_midi and gap <= max_gap:
                result[-1] = NoteEvent(
                    pitch_midi=last.pitch_midi,
                    start_time=last.start_time,
                    end_time=note.end_time,
                    velocity=max(last.velocity, note.velocity),
                    confidence=max(last.confidence, note.confidence),
                )
            else:
                result.append(note)
        return result

    def _filter_short(self, notes: List[NoteEvent],
                      min_dur: float = 0.06) -> List[NoteEvent]:
        return [n for n in notes if n.duration >= min_dur]
