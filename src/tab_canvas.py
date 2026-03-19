"""
그래픽 TAB 캔버스 위젯
tkinter Canvas를 이용해 악보 형태의 베이스 TAB을 렌더링합니다.

레이아웃:
  ┌─ 행 ────────────────────────────────────────────┐
  │ G ─────────[5]─────────[7]────────────────────  │
  │ D ──────────────────────────────────────────── │
  │ A ──[3]──────────────────────────[5]────────── │
  │ E ──────────────────────────────────────────── │
  │     | 마디 1        | 마디 2        | 마디 3   │
  └────────────────────────────────────────────────┘
"""
from __future__ import annotations

import tkinter as tk
from typing import List, Optional

from .fretboard_mapper import TabNote


class TabCanvas(tk.Frame):
    # ── 레이아웃 상수 ────────────────────────────────────────────────
    PX_PER_BEAT   = 56      # 1박 기본 픽셀 (동적으로 조정됨)
    STR_SPACING   = 46      # 현 간격 (px)
    STR_TOP_PAD   = 52      # 행 상단 ↔ G현 거리
    ROW_BOTTOM    = 44      # E현 ↔ 다음 행 거리
    LEFT_MARGIN   = 54      # 현 이름 영역
    RIGHT_MARGIN  = 28
    BEATS_PER_BAR = 4

    # ── 색상 팔레트 ──────────────────────────────────────────────────
    BG         = '#1A1A2E'
    STRING_CLR = '#3A3A5C'     # 현 라인
    BAR_CLR    = '#55557A'     # 마디선
    EDGE_CLR   = '#9090B8'     # 행 경계선
    MNUM_CLR   = '#606080'     # 마디 번호
    NAME_CLR   = '#9999BB'     # 현 이름

    # 현별 노트 텍스트 색 (밝은 색으로 현 선 위에서 가독성 확보)
    NOTE_COLORS  = ['#42A5F5', '#66BB6A', '#FF9800', '#CE93D8']  # G D A E

    # 노트 간 최소 픽셀 간격 (동적 PX_PER_BEAT 계산 기준)
    MIN_NOTE_GAP_PX = 22

    FONT_NOTE    = ('Helvetica', 11, 'bold')
    FONT_STR     = ('Helvetica', 11, 'bold')
    FONT_MNUM    = ('Helvetica', 9)

    STRING_NAMES = ['G', 'D', 'A', 'E']

    def __init__(self, master, **kw):
        kw.setdefault('bg', self.BG)
        super().__init__(master, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._cv = tk.Canvas(self, bg=self.BG, bd=0, highlightthickness=0)
        vsb = tk.Scrollbar(self, orient='vertical',   command=self._cv.yview)
        hsb = tk.Scrollbar(self, orient='horizontal', command=self._cv.xview)
        self._cv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._cv.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        # 마우스 휠 스크롤
        self._cv.bind('<MouseWheel>',
                      lambda e: self._cv.yview_scroll(-1 * (e.delta // 120), 'units'))

        self._notes:        List[TabNote] = []
        self._bpm:          float = 120.0
        self._bpr:          int   = 4
        self._dyn_px_beat:  int   = self.PX_PER_BEAT  # 동적 계산값

    # ── 공개 API ─────────────────────────────────────────────────────
    def set_notes(self, notes: List[TabNote], bpm: float = 120.0,
                  bars_per_row: int = 4):
        self._notes = notes
        self._bpm   = max(1.0, bpm)
        self._bpr   = max(1, bars_per_row)
        self._dyn_px_beat = self._compute_px_per_beat(notes, self._bpm)
        self.redraw()

    def clear(self):
        self._notes = []
        self._cv.delete('all')
        self._cv.configure(scrollregion=(0, 0, 100, 100))

    # ── 렌더링 ───────────────────────────────────────────────────────
    def _compute_px_per_beat(self, notes: List[TabNote], bpm: float) -> int:
        """노트 간 최소 간격을 기준으로 PX_PER_BEAT를 동적으로 결정."""
        if len(notes) < 2:
            return self.PX_PER_BEAT
        beat_dur = 60.0 / bpm
        sorted_beats = sorted(n.start_time * (bpm / 60.0) for n in notes)
        gaps = [b - a for a, b in zip(sorted_beats, sorted_beats[1:]) if b - a > 1e-4]
        if not gaps:
            return self.PX_PER_BEAT
        min_gap = min(gaps)
        # min_gap 비트 간격이 MIN_NOTE_GAP_PX 이상이 되도록 px_per_beat 확장
        needed = int(self.MIN_NOTE_GAP_PX / max(min_gap, 0.01))
        return max(self.PX_PER_BEAT, min(needed, 300))

    @property
    def _row_h(self) -> int:
        return self.STR_TOP_PAD + 3 * self.STR_SPACING + self.ROW_BOTTOM

    @property
    def _bar_w(self) -> int:
        return self._dyn_px_beat * self.BEATS_PER_BAR

    def redraw(self):
        cv = self._cv
        cv.delete('all')

        if not self._notes:
            cv.create_text(
                300, 120,
                text='분석된 음표가 없습니다.',
                fill='#666688', font=('Helvetica', 14),
            )
            return

        bpm          = self._bpm
        bpr          = self._bpr
        beats_per_row = bpr * self.BEATS_PER_BAR
        total_sec    = max(n.end_time for n in self._notes)
        total_beats  = total_sec * (bpm / 60.0)
        num_rows     = max(1, int(total_beats / beats_per_row) + 1)

        rh    = self._row_h
        bw    = self._bar_w
        row_w = self.LEFT_MARGIN + bpr * bw + self.RIGHT_MARGIN
        total_h = num_rows * rh + 20

        cv.configure(scrollregion=(0, 0, row_w, total_h))

        # 행 배경 + 현 + 마디선
        for row in range(num_rows):
            self._draw_row(cv, row, row * rh, bpr, bw, row_w)

        # 음표
        for note in self._notes:
            t_beats     = note.start_time * (bpm / 60.0)
            row         = int(t_beats / beats_per_row)
            beat_in_row = t_beats % beats_per_row
            if row >= num_rows:
                continue
            cx = self.LEFT_MARGIN + beat_in_row * self._dyn_px_beat
            cy = row * rh + self.STR_TOP_PAD + note.string_idx * self.STR_SPACING
            self._draw_note(cv, cx, cy, note.fret, note.string_idx)

    def _draw_row(self, cv, row: int, y0: int, bpr: int, bw: int, row_w: int):
        # 현 라인 + 이름
        for s in range(4):
            sy = y0 + self.STR_TOP_PAD + s * self.STR_SPACING
            x0 = self.LEFT_MARGIN
            x1 = x0 + bpr * bw
            # 현 이름
            cv.create_text(
                x0 - 10, sy,
                text=self.STRING_NAMES[s],
                fill=self.NAME_CLR,
                font=self.FONT_STR,
                anchor='e',
            )
            # 현 라인 (굵기: G·E는 조금 더 눈에 띄게)
            lw = 2 if s in (0, 3) else 1
            cv.create_line(x0, sy, x1, sy, fill=self.STRING_CLR, width=lw)

        # 마디선
        sy_top = y0 + self.STR_TOP_PAD
        sy_bot = y0 + self.STR_TOP_PAD + 3 * self.STR_SPACING
        for b in range(bpr + 1):
            x     = self.LEFT_MARGIN + b * bw
            is_edge = (b == 0 or b == bpr)
            color = self.EDGE_CLR if is_edge else self.BAR_CLR
            lw    = 2 if is_edge else 1
            cv.create_line(x, sy_top, x, sy_bot, fill=color, width=lw)
            # 마디 번호
            if b < bpr:
                mnum = row * bpr + b + 1
                cv.create_text(
                    x + 4, sy_bot + 16,
                    text=str(mnum),
                    fill=self.MNUM_CLR,
                    font=self.FONT_MNUM,
                    anchor='w',
                )

        # 행 구분선 (첫 행 제외)
        if row > 0:
            cv.create_line(0, y0, row_w, y0, fill='#262640', width=1, dash=(4, 8))

    def _draw_note(self, cv, cx: float, cy: float, fret: int, string_idx: int):
        fstr = str(fret)
        color = self.NOTE_COLORS[string_idx]

        # 현 선 위 텍스트 영역을 BG색으로 지워 현 선이 숫자를 가리지 않게 함
        tw = len(fstr) * 8 + 6
        th = 14
        cv.create_rectangle(
            int(cx - tw / 2), int(cy - th / 2),
            int(cx + tw / 2), int(cy + th / 2),
            fill=self.BG, outline='',
        )
        # 프렛 번호 텍스트만 표시
        cv.create_text(
            int(cx), int(cy),
            text=fstr,
            fill=color,
            font=self.FONT_NOTE,
        )

    # ── 이미지로 내보내기 (PIL 필요) ─────────────────────────────────
    def export_image(self, path: str):
        """현재 전체 TAB을 PNG 이미지로 저장."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            raise RuntimeError("PIL이 필요합니다: pip install pillow")

        if not self._notes:
            raise ValueError("음표가 없습니다.")

        bpm           = self._bpm
        bpr           = self._bpr
        px_beat       = self._dyn_px_beat
        beats_per_row = bpr * self.BEATS_PER_BAR
        total_sec     = max(n.end_time for n in self._notes)
        total_beats   = total_sec * (bpm / 60.0)
        num_rows      = max(1, int(total_beats / beats_per_row) + 1)

        rh      = self._row_h
        bw      = px_beat * self.BEATS_PER_BAR
        row_w   = self.LEFT_MARGIN + bpr * bw + self.RIGHT_MARGIN
        total_h = num_rows * rh + 20

        img  = Image.new('RGB', (row_w, total_h), color=self.BG)
        draw = ImageDraw.Draw(img)

        def hex2rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        bg_rgb = hex2rgb(self.BG)

        for row in range(num_rows):
            y0   = row * rh
            sy_t = y0 + self.STR_TOP_PAD
            sy_b = y0 + self.STR_TOP_PAD + 3 * self.STR_SPACING

            # 현 라인
            for s in range(4):
                sy = y0 + self.STR_TOP_PAD + s * self.STR_SPACING
                draw.line([(self.LEFT_MARGIN, sy),
                            (self.LEFT_MARGIN + bpr * bw, sy)],
                           fill=hex2rgb(self.STRING_CLR), width=1)
                draw.text((self.LEFT_MARGIN - 14, sy - 6),
                           self.STRING_NAMES[s],
                           fill=hex2rgb(self.NAME_CLR))

            # 마디선
            for b in range(bpr + 1):
                x = self.LEFT_MARGIN + b * bw
                draw.line([(x, sy_t), (x, sy_b)],
                           fill=hex2rgb(self.BAR_CLR), width=1)

        # 음표 — 박스 없이 텍스트만
        for note in self._notes:
            tb  = note.start_time * (bpm / 60.0)
            row = int(tb / beats_per_row)
            br  = tb % beats_per_row
            if row >= num_rows:
                continue
            cx  = self.LEFT_MARGIN + br * px_beat
            cy  = row * rh + self.STR_TOP_PAD + note.string_idx * self.STR_SPACING
            fstr = str(note.fret)
            tw = len(fstr) * 8 + 6
            th = 14
            x0 = int(cx - tw / 2)
            y0 = int(cy - th / 2)
            # BG 클리어 (현 선 위)
            draw.rectangle([(x0, y0), (x0 + tw, y0 + th)], fill=bg_rgb)
            # 텍스트
            draw.text((x0 + 2, y0 + 1), fstr,
                       fill=hex2rgb(self.NOTE_COLORS[note.string_idx]))

        img.save(path)
