"""
TAB rendering module.

Outputs ASCII tablature with optional measure bars based on BPM.

TAB format example (standard):
  G|---5---7---------|---5---7---------|
  D|-------5---7-----|-------5---7-----|
  A|-3---3---3---3---|                  |
  E|-----------------|-----------------|
"""
from __future__ import annotations

from typing import List

import numpy as np

from .fretboard_mapper import TabNote


# ── 조성 감지 (Krumhansl-Kessler 피치 클래스 프로파일) ──────────────────
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_key(tab_notes: List[TabNote]) -> str:
    """
    음표의 피치 클래스 히스토그램을 Krumhansl-Kessler 알고리즘으로 분석해
    조성(장조/단조)을 판별합니다.
    """
    if not tab_notes:
        return ''

    # 음표 길이 가중 피치 클래스 히스토그램
    hist = np.zeros(12)
    for n in tab_notes:
        pc = n.pitch_midi % 12
        hist[pc] += max(n.end_time - n.start_time, 0.01)

    if hist.sum() == 0:
        return ''
    hist = hist / hist.sum()

    best_corr = -np.inf
    best_key  = ''

    for root in range(12):
        for profile, mode in [(_MAJOR, '장조'), (_MINOR, '단조')]:
            rotated = np.array([profile[(i - root) % 12] for i in range(12)])
            corr    = float(np.corrcoef(hist, rotated)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_key  = f"Key: {_NOTE_NAMES[root]} {mode}"

    return best_key


class TabRenderer:
    """Renders a list of TabNote objects as ASCII tablature."""

    # ------------------------------------------------------------------
    def render_measures(self,
                        tab_notes: List[TabNote],
                        bpm: float = 120.0,
                        beats_per_measure: int = 4,
                        measures_per_row: int = 4,
                        resolution: int = 8) -> str:
        """
        Render TAB aligned to a musical grid.

        Parameters
        ----------
        tab_notes        : list of TabNote
        bpm              : tempo in beats per minute
        beats_per_measure: time signature numerator (default 4 → 4/4)
        measures_per_row : how many measures to print per line
        resolution       : grid subdivisions per beat (16 → 16th notes)
        """
        if not tab_notes:
            return "No notes detected."

        num_strings = 4
        beat_dur = 60.0 / bpm
        subdiv_dur = beat_dur / resolution           # duration of one grid cell
        cols_per_measure = beats_per_measure * resolution   # e.g. 4*16 = 64

        total_dur = max(n.end_time for n in tab_notes) + beat_dur * beats_per_measure
        num_measures = max(1, int(total_dur / (beat_dur * beats_per_measure)) + 1)
        total_cols = num_measures * cols_per_measure

        # ---- Build grid ----
        # grid[string_idx][col] = character (default '-')
        grid: List[List[str]] = [['-'] * total_cols for _ in range(num_strings)]

        for note in tab_notes:
            col = int(round(note.start_time / subdiv_dur))
            col = min(col, total_cols - 1)
            fret_str = str(note.fret)
            for k, ch in enumerate(fret_str):
                if col + k < total_cols:
                    grid[note.string_idx][col + k] = ch
                    # Ensure adjacent cells for multi-digit frets are marked
                    # (next note can't start in the middle of a fret number)

        # ---- Render rows ----
        string_names = ['G', 'D', 'A', 'E']
        if hasattr(tab_notes[0], 'string_name'):
            pass  # could customise, not needed for 4-string

        lines: List[str] = []

        for row_m in range(0, num_measures, measures_per_row):
            end_m = min(row_m + measures_per_row, num_measures)

            for s in range(num_strings):
                name = string_names[s]
                row = f"{name}|"
                for m in range(row_m, end_m):
                    c0 = m * cols_per_measure
                    c1 = c0 + cols_per_measure
                    row += ''.join(grid[s][c0:c1]) + '|'
                lines.append(row)

            # Measure number annotation
            measure_nums = '  '
            for m in range(row_m, end_m):
                label = str(m + 1)
                measure_nums += label.ljust(cols_per_measure + 1)
            lines.append(measure_nums.rstrip())
            lines.append('')

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def render_simple(self,
                      tab_notes: List[TabNote],
                      columns_per_row: int = 80,
                      time_scale: float = 10.0) -> str:
        """
        Simple time-proportional TAB without measure bars.
        Useful for rubato or unmeasured passages.

        time_scale : columns per second
        """
        if not tab_notes:
            return "No notes detected."

        num_strings = 4
        total_dur = max(n.end_time for n in tab_notes) + 1.0
        total_cols = max(int(total_dur * time_scale) + 1, 1)

        grid: List[List[str]] = [['-'] * total_cols for _ in range(num_strings)]

        for note in tab_notes:
            col = int(note.start_time * time_scale)
            col = min(col, total_cols - 1)
            fret_str = str(note.fret)
            for k, ch in enumerate(fret_str):
                if col + k < total_cols:
                    grid[note.string_idx][col + k] = ch

        string_names = ['G', 'D', 'A', 'E']
        lines: List[str] = []

        for row_start in range(0, total_cols, columns_per_row):
            row_end = min(row_start + columns_per_row, total_cols)
            for s in range(num_strings):
                name = string_names[s]
                content = ''.join(grid[s][row_start:row_end])
                lines.append(f"{name}|{content}|")
            lines.append('')

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def note_list(self, tab_notes: List[TabNote]) -> str:
        """Human-readable list of all detected notes (for debugging)."""
        if not tab_notes:
            return "No notes."
        import librosa
        rows = ["  #   Time(s)   End(s)    String  Fret  Note"]
        rows.append("  " + "-" * 50)
        for i, n in enumerate(tab_notes):
            note_name = librosa.midi_to_note(n.pitch_midi)
            rows.append(
                f"  {i+1:3d}  {n.start_time:6.2f}   {n.end_time:6.2f}"
                f"    {n.string_name}       {n.fret:2d}    {note_name}"
            )
        return '\n'.join(rows)
