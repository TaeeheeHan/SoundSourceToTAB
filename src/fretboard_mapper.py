"""
Fretboard mapper: converts MIDI note events to bass TAB positions.

Strategy: greedy algorithm that minimises total hand-position movement,
with bonuses for open strings and a preference for middle strings (A/D)
which are the most natural positions for bass lines.

Standard 4-string bass tuning (MIDI note numbers):
  String index  Name  Open MIDI
  0             G     43  (G2)
  1             D     38  (D2)
  2             A     33  (A1)
  3             E     28  (E1)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .pitch_detector import NoteEvent


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class TabNote:
    string_idx: int     # 0=G (top of TAB), 1=D, 2=A, 3=E (bottom)
    fret: int
    start_time: float
    end_time: float
    pitch_midi: int

    @property
    def string_name(self) -> str:
        return STRING_NAMES[self.string_idx]


# ------------------------------------------------------------------
# Tuning presets
# ------------------------------------------------------------------

STRING_NAMES = ['G', 'D', 'A', 'E']

TUNINGS = {
    'standard':    [43, 38, 33, 28],   # G D A E
    'drop_d':      [43, 38, 33, 26],   # G D A D  (low D)
    'standard_5':  [43, 38, 33, 28, 23],  # G D A E B (5-string)
    'drop_d_5':    [43, 38, 33, 28, 21],  # G D A E A (5-string drop)
}


# ------------------------------------------------------------------
# Mapper
# ------------------------------------------------------------------

class FretboardMapper:
    MAX_FRET = 24

    def __init__(self, tuning: str | List[int] = 'standard'):
        if isinstance(tuning, str):
            self.open_notes = TUNINGS.get(tuning, TUNINGS['standard'])
        else:
            self.open_notes = list(tuning)

        self.num_strings = len(self.open_notes)
        # Build string names (extend with numbered names if 5-string)
        if self.num_strings == 4:
            self.string_names = STRING_NAMES
        else:
            self.string_names = ['G', 'D', 'A', 'E', 'B'][:self.num_strings]

    # ------------------------------------------------------------------
    def get_positions(self, midi_pitch: int) -> List[Tuple[int, int]]:
        """Return all (string_idx, fret) positions that produce this pitch."""
        positions = []
        for s, open_midi in enumerate(self.open_notes):
            fret = midi_pitch - open_midi
            if 0 <= fret <= self.MAX_FRET:
                positions.append((s, fret))
        return positions

    # ------------------------------------------------------------------
    def map_notes(self, note_events: List[NoteEvent]) -> List[TabNote]:
        """
        Greedy fretboard mapping that minimises hand movement.
        Returns a list of TabNote in chronological order.
        """
        if not note_events:
            return []

        tab_notes: List[TabNote] = []
        # Start hand position at fret 2 (low position, common for bass)
        hand_fret: float = 2.0

        for note in sorted(note_events, key=lambda n: n.start_time):
            positions = self.get_positions(note.pitch_midi)
            if not positions:
                # Out of range — skip
                continue

            s, fret = self._best_position(positions, hand_fret)

            tab_notes.append(TabNote(
                string_idx=s,
                fret=fret,
                start_time=note.start_time,
                end_time=note.end_time,
                pitch_midi=note.pitch_midi,
            ))

            # Only non-open frets move the hand
            if fret > 0:
                hand_fret = float(fret)

        return tab_notes

    # ------------------------------------------------------------------
    def _best_position(self,
                       positions: List[Tuple[int, int]],
                       hand_fret: float) -> Tuple[int, int]:
        """
        2단계 선택:
          1단계: 이동 거리(+개방현 보너스) 기준으로 최솟값 산출
          2단계: 최솟값 +2 이내 후보들 중에서만 현 우선순위 적용
                 → 멀리 있는 E/A 포지션이 가까운 D/G 포지션을 이기지 못함
        """
        def dist_cost(pos: Tuple[int, int]) -> float:
            s, fret = pos
            return abs(fret - hand_fret) + fret * 0.06 + (-2.0 if fret == 0 else 0.0)

        def string_rank(pos: Tuple[int, int]) -> int:
            """낮을수록 우선 (12프렛 이내 E·A > D·G > 12프렛 초과 E·A > D·G)"""
            s, fret = pos
            is_ea = s in (2, 3)
            in_12 = fret <= 12
            if   is_ea and in_12:     return 0
            elif not is_ea and in_12: return 1
            elif is_ea:               return 2
            else:                     return 3

        min_dc    = min(dist_cost(p) for p in positions)
        # 거리상 2 이내인 후보만 현 우선순위 대상으로 제한
        TOLERANCE = 2.0
        candidates = [p for p in positions if dist_cost(p) <= min_dc + TOLERANCE]

        # 우선순위 먼저, 동률이면 거리로 결정
        return min(candidates, key=lambda p: (string_rank(p), dist_cost(p)))
