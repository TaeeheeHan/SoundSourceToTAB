"""
Bass Tab Transcriber  —  메인 GUI (한국어)
CustomTkinter + 그래픽 TAB 캔버스
"""
from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import customtkinter as ctk

from .audio_processor import AudioProcessor, SUPPORTED_FORMATS
from .pitch_detector import PitchDetector, _basic_pitch_available, _crepe_available
from .fretboard_mapper import FretboardMapper, TUNINGS
from .tab_renderer import TabRenderer, detect_key, detect_key_from_audio
from .tab_canvas import TabCanvas

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ──────────────────────────────────────────────────────────────────────────────
class BassTabApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("베이스 TAB 트랜스크라이버")
        self.geometry("1280x820")
        self.minsize(960, 650)

        # 핵심 객체
        self.audio_proc   = AudioProcessor()
        self.detector     = PitchDetector(method='auto')
        self.mapper       = FretboardMapper('standard')
        self.renderer     = TabRenderer()

        # 상태
        self.current_file: str | None = None
        self.tab_notes    = None
        self.note_events  = None

        self._build_ui()
        self._try_drag_drop()

    # ─────────────────────────── UI 구성 ─────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_toolbar()
        self._build_body()

    # ── 툴바 ─────────────────────────────────────────────────────────
    def _build_toolbar(self):
        bar = ctk.CTkFrame(self, height=56, corner_radius=0)
        bar.grid(row=0, column=0, sticky='ew')
        bar.grid_columnconfigure(8, weight=1)   # 빈 공간

        ctk.CTkLabel(bar, text='🎸 베이스 TAB 트랜스크라이버',
                     font=ctk.CTkFont(size=15, weight='bold')).grid(
            row=0, column=0, padx=14, pady=8)

        self.btn_open = ctk.CTkButton(bar, text='파일 열기', width=100,
                                      command=self.open_file)
        self.btn_open.grid(row=0, column=1, padx=4)

        self.btn_analyze = ctk.CTkButton(bar, text='분석', width=90,
                                          state='disabled',
                                          command=self.start_analysis,
                                          fg_color='#1565C0',
                                          hover_color='#1976D2')
        self.btn_analyze.grid(row=0, column=2, padx=4)

        self.btn_export = ctk.CTkButton(bar, text='저장', width=90,
                                         state='disabled',
                                         command=self.export_tab,
                                         fg_color='#2E7D32',
                                         hover_color='#388E3C')
        self.btn_export.grid(row=0, column=3, padx=4)

        # 구분
        ctk.CTkLabel(bar, text='│', text_color='#444').grid(
            row=0, column=9, padx=8)

        # BPM
        ctk.CTkLabel(bar, text='BPM:').grid(row=0, column=10, padx=(4, 2))
        self.bpm_var = ctk.StringVar(value='120')
        ctk.CTkEntry(bar, textvariable=self.bpm_var, width=56).grid(
            row=0, column=11, padx=(0, 4))

        self.btn_bpm_auto = ctk.CTkButton(bar, text='BPM 자동', width=82,
                                           state='disabled',
                                           command=self.detect_bpm_auto,
                                           fg_color='#4A148C',
                                           hover_color='#6A1B9A')
        self.btn_bpm_auto.grid(row=0, column=12, padx=(0, 8))

        # 마디 오프셋
        ctk.CTkLabel(bar, text='시작 마디:').grid(row=0, column=15, padx=(4, 2))
        self.measure_offset_var = ctk.StringVar(value='0')
        offset_entry = ctk.CTkEntry(bar, textvariable=self.measure_offset_var, width=44)
        offset_entry.grid(row=0, column=16, padx=(0, 8))
        self.measure_offset_var.trace_add('write', lambda *_: self._on_offset_change())

        # 화면 모드
        ctk.CTkLabel(bar, text='보기:').grid(row=0, column=13, padx=(4, 2))
        self.view_var = ctk.StringVar(value='악보')
        ctk.CTkOptionMenu(bar, variable=self.view_var,
                          values=['악보', '음표 목록', '텍스트 TAB'],
                          width=120,
                          command=self._on_view_change).grid(
            row=0, column=14, padx=(0, 12))

    # ── 바디 ─────────────────────────────────────────────────────────
    def _build_body(self):
        body = ctk.CTkFrame(self, corner_radius=0, fg_color='transparent')
        body.grid(row=1, column=0, sticky='nsew', padx=8, pady=8)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # ── 왼쪽 패널 ─────────────────────────────────────────────────
        left = ctk.CTkFrame(body, width=272, corner_radius=8)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)

        # 파일 드롭 영역
        self.drop_frame = ctk.CTkFrame(left, height=130, corner_radius=8,
                                        border_width=2, border_color='#444')
        self.drop_frame.grid(row=0, column=0, padx=12, pady=(12, 6), sticky='ew')
        self.drop_frame.grid_propagate(False)
        self.drop_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.drop_frame, text='파일을 여기에 드래그',
                     font=ctk.CTkFont(size=13, weight='bold')).grid(
            row=0, column=0, pady=(18, 2))
        ctk.CTkLabel(self.drop_frame, text='MP3 / WAV / FLAC',
                     text_color='gray', font=ctk.CTkFont(size=11)).grid(
            row=1, column=0)

        self.file_label = ctk.CTkLabel(self.drop_frame, text='',
                                        wraplength=240,
                                        text_color='#4CAF50',
                                        font=ctk.CTkFont(size=10))
        self.file_label.grid(row=2, column=0, padx=8, pady=(4, 14))

        for w in [self.drop_frame] + list(self.drop_frame.winfo_children()):
            w.bind('<Button-1>', lambda _: self.open_file())

        # 상태 + 진행
        self.status_lbl = ctk.CTkLabel(left, text='준비',
                                        text_color='gray',
                                        font=ctk.CTkFont(size=11))
        self.status_lbl.grid(row=1, column=0, pady=(4, 0))

        self.progress = ctk.CTkProgressBar(left)
        self.progress.grid(row=2, column=0, padx=14, pady=4, sticky='ew')
        self.progress.set(0)

        # ── 설정 ─────────────────────────────────────────────────────
        cfg = ctk.CTkFrame(left, corner_radius=8)
        cfg.grid(row=3, column=0, padx=12, pady=6, sticky='ew')
        cfg.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(cfg, text='설정',
                     font=ctk.CTkFont(weight='bold')).grid(
            row=0, column=0, columnspan=2, padx=10, pady=(10, 4), sticky='w')

        # AI 사용 여부 표시
        if _crepe_available():
            ai_txt = '✓ CREPE (딥러닝)'
            ai_clr = '#00BCD4'
        elif _basic_pitch_available():
            ai_txt = '✓ AI 모델 (Basic Pitch)'
            ai_clr = '#4CAF50'
        else:
            ai_txt = '○ Librosa (AI 없음)'
            ai_clr = '#FF9800'
        ctk.CTkLabel(cfg, text=ai_txt, text_color=ai_clr,
                     font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, columnspan=2, padx=10, pady=(0, 6), sticky='w')

        def _row(parent, r, label, widget_fn):
            ctk.CTkLabel(parent, text=label).grid(
                row=r, column=0, padx=10, pady=4, sticky='w')
            w = widget_fn(parent)
            w.grid(row=r, column=1, padx=8, pady=4)

        # 감지 방법
        ctk.CTkLabel(cfg, text='감지 방법:').grid(
            row=2, column=0, padx=10, pady=4, sticky='w')
        self.method_var = ctk.StringVar(value='auto')
        ctk.CTkOptionMenu(cfg, variable=self.method_var,
                          values=['auto', 'librosa', 'basic-pitch', 'crepe'],
                          width=120).grid(row=2, column=1, padx=8, pady=4)

        # 조율
        ctk.CTkLabel(cfg, text='조율:').grid(
            row=3, column=0, padx=10, pady=4, sticky='w')
        self.tuning_var = ctk.StringVar(value='standard')
        ctk.CTkOptionMenu(cfg, variable=self.tuning_var,
                          values=list(TUNINGS.keys()),
                          width=120).grid(row=3, column=1, padx=8, pady=4)

        # 신뢰도 임계값
        ctk.CTkLabel(cfg, text='신뢰도:').grid(
            row=4, column=0, padx=10, pady=4, sticky='w')
        self.conf_var = ctk.DoubleVar(value=0.25)
        ctk.CTkSlider(cfg, from_=0.0, to=1.0,
                      variable=self.conf_var, width=120).grid(
            row=4, column=1, padx=8, pady=4)
        self.conf_lbl = ctk.CTkLabel(cfg, text='0.25',
                                      font=ctk.CTkFont(size=10))
        self.conf_lbl.grid(row=5, column=1, padx=8, pady=(0, 2))
        self.conf_var.trace_add('write',
            lambda *_: self.conf_lbl.configure(
                text=f'{self.conf_var.get():.2f}'))

        # 마디/행
        ctk.CTkLabel(cfg, text='마디/행:').grid(
            row=6, column=0, padx=10, pady=4, sticky='w')
        self.bpr_var = ctk.StringVar(value='4')
        ctk.CTkOptionMenu(cfg, variable=self.bpr_var,
                          values=['2', '3', '4', '6', '8'],
                          width=120).grid(row=6, column=1, padx=8,
                                          pady=(4, 12))

        # ── 오른쪽: TAB 표시 영역 ──────────────────────────────────
        right = ctk.CTkFrame(body, corner_radius=8)
        right.grid(row=0, column=1, sticky='nsew')
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        # 제목 바
        title_bar = ctk.CTkFrame(right, height=36, fg_color='transparent')
        title_bar.grid(row=0, column=0, sticky='ew', padx=10, pady=(6, 0))
        title_bar.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(title_bar, text='타브 악보',
                     font=ctk.CTkFont(size=14, weight='bold')).grid(
            row=0, column=0, sticky='w')

        self.note_count_lbl = ctk.CTkLabel(title_bar, text='',
                                            text_color='gray',
                                            font=ctk.CTkFont(size=11))
        self.note_count_lbl.grid(row=0, column=1, padx=12, sticky='w')

        # ── TAB 캔버스 (그래픽) ───────────────────────────────────
        self.tab_canvas = TabCanvas(right, bg='#1A1A2E')
        self.tab_canvas.grid(row=1, column=0, sticky='nsew',
                              padx=6, pady=(4, 4))

        # ── 텍스트 TAB (숨김 초기) ───────────────────────────────
        self.tab_text = ctk.CTkTextbox(
            right,
            font=ctk.CTkFont(family='Courier New', size=12),
            wrap='none',
        )
        # 처음엔 숨김 (악보 모드가 기본)
        self.tab_text.grid(row=1, column=0, sticky='nsew', padx=6, pady=(4, 4))
        self.tab_text.grid_remove()

        self._set_tab_text(
            '베이스 오디오 파일을 열어주세요.\n\n'
            '지원 형식: MP3, WAV, FLAC\n\n'
            '파일을 드래그하거나 "파일 열기" 버튼을 클릭하세요.'
        )

        # 초기에는 악보(캔버스) 보이기
        self._show_canvas(True)

    # ─────────────────────────── 드래그 & 드롭 ───────────────────────

    def _try_drag_drop(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            pass

    def _on_drop(self, event):
        path = event.data.strip().lstrip('{').rstrip('}')
        self.load_file(path)

    # ─────────────────────────── 마디 오프셋 ────────────────────────

    def _get_measure_offset(self) -> int:
        try:
            return int(self.measure_offset_var.get())
        except (ValueError, AttributeError):
            return 0

    def _on_offset_change(self):
        """오프셋 변경 즉시 악보/텍스트 TAB 재렌더링."""
        if not self.tab_notes:
            return
        try:
            bpm = float(self.bpm_var.get())
        except ValueError:
            bpm = 120.0
        bpr    = int(self.bpr_var.get())
        offset = self._get_measure_offset()
        # 텍스트 TAB 재렌더
        tab_text = self.renderer.render_measures(
            self.tab_notes, bpm=bpm, measures_per_row=bpr,
            measure_offset=offset)
        header = (
            f'파일: {Path(self.current_file).name}  |  '
            f'BPM: {bpm}  |  {self.tab_canvas._key}\n'
            f'{"─" * 60}\n\n'
        )
        self._set_tab_text(header + tab_text)
        # 캔버스는 마디 번호 오프셋 반영
        self.tab_canvas.set_notes(self.tab_notes, bpm=bpm,
                                   bars_per_row=bpr,
                                   key=self.tab_canvas._key,
                                   measure_offset=offset)

    # ─────────────────────────── BPM 자동 감지 ──────────────────────

    def detect_bpm_auto(self):
        """librosa로 BPM을 자동 감지해 입력창에 채워 넣는다."""
        if not self.current_file:
            return
        self.btn_bpm_auto.configure(state='disabled', text='분석 중…')
        self.status_lbl.configure(text='BPM 감지 중…', text_color='orange')

        def _worker():
            try:
                import librosa as _lb
                y, sr = _lb.load(self.current_file, sr=22050, mono=True)
                tempo, _ = _lb.beat.beat_track(y=y, sr=sr)
                bpm = int(round(float(tempo)))
                self._ui(lambda: self._on_bpm_done(bpm))
            except Exception as exc:
                self._ui(lambda: self._on_bpm_error(str(exc)))

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def _on_bpm_done(self, bpm: int):
        self.bpm_var.set(str(bpm))
        self.btn_bpm_auto.configure(state='normal', text='BPM 자동')
        self.status_lbl.configure(text=f'BPM 감지 완료: {bpm}', text_color='#4CAF50')

    def _on_bpm_error(self, msg: str):
        self.btn_bpm_auto.configure(state='normal', text='BPM 자동')
        self.status_lbl.configure(text='BPM 감지 실패', text_color='red')

    # ─────────────────────────── 파일 처리 ───────────────────────────

    def open_file(self):
        exts = ' '.join(f'*{e}' for e in SUPPORTED_FORMATS)
        path = filedialog.askopenfilename(
            title='베이스 오디오 파일 선택',
            filetypes=[('오디오 파일', exts), ('모든 파일', '*.*')],
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str):
        self.current_file = path
        self.tab_notes    = None
        self.note_events  = None
        name = Path(path).name
        self.file_label.configure(text=name)
        self.btn_analyze.configure(state='normal')
        self.btn_bpm_auto.configure(state='normal')
        self.btn_export.configure(state='disabled')
        self.status_lbl.configure(text='분석 준비됨', text_color='#4CAF50')
        self.progress.set(0)
        self.tab_canvas.clear()
        self._set_tab_text(f'파일 로드: {name}\n\n"분석" 버튼을 클릭하세요.')

    # ─────────────────────────── 분석 ────────────────────────────────

    def start_analysis(self):
        if not self.current_file:
            return

        self.btn_analyze.configure(state='disabled')
        self.btn_export.configure(state='disabled')
        self.status_lbl.configure(text='분석 중…', text_color='orange')
        self.progress.set(0)
        self.note_count_lbl.configure(text='')

        self.detector = PitchDetector(method=self.method_var.get())
        self.mapper   = FretboardMapper(self.tuning_var.get())

        threading.Thread(target=self._analysis_thread, daemon=True).start()

    def _analysis_thread(self):
        import pathlib
        _log = pathlib.Path.home() / 'Desktop' / 'Bass' / 'debug.log'
        def _write(msg):
            with open(_log, 'a', encoding='utf-8') as f:
                import datetime
                f.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")
        try:
            _write(f"분석 시작: {self.current_file}")
            self._ui(lambda: self._set_tab_text('오디오 파일 로딩 중…'))
            self._ui(lambda: self.progress.set(0.1))

            self._ui(lambda: self._set_tab_text(
                '음정 감지 중…\n\n'
                'AI 모델을 사용 중이면 처음에 30–60초 정도 걸릴 수 있습니다.'))
            self._ui(lambda: self.progress.set(0.2))

            min_conf = self.conf_var.get()
            _write(f"음정 감지 시작 (method={self.method_var.get()}, conf={min_conf})")
            note_events = self.detector.detect_notes(
                self.current_file, min_confidence=min_conf)
            self.note_events = note_events
            _write(f"음정 감지 완료: {len(note_events)}개")

            self._ui(lambda: self.progress.set(0.65))
            self._ui(lambda: self._set_tab_text(
                f'{len(note_events)}개 음표 감지 완료.\n프렛보드 매핑 중…'))

            tab_notes = self.mapper.map_notes(note_events)
            self.tab_notes = tab_notes

            self._ui(lambda: self.progress.set(0.85))

            try:
                bpm = float(self.bpm_var.get())
            except ValueError:
                bpm = 120.0
            bpr = int(self.bpr_var.get())

            # 오디오 크로마 기반 조성 감지 (음표 히스토그램보다 정확)
            key_str = detect_key_from_audio(self.current_file)
            if not key_str:
                key_str = detect_key(tab_notes)   # 폴백

            offset = self._get_measure_offset()
            tab_text = self.renderer.render_measures(
                tab_notes, bpm=bpm, measures_per_row=bpr,
                measure_offset=offset)
            header = (
                f'파일: {Path(self.current_file).name}  |  '
                f'BPM: {bpm}  |  '
                f'음표: {len(note_events)}개  |  {key_str}\n'
                f'{"─" * 60}\n\n'
            )

            self._ui(lambda: self.progress.set(1.0))
            self._ui(lambda: self._on_done(tab_notes, bpm, bpr,
                                            header + tab_text, len(note_events),
                                            key_str, offset))

        except Exception:
            err = traceback.format_exc()
            _write(f"에러 발생:\n{err}")
            self._ui(lambda: self._on_error(err))

    def _on_done(self, tab_notes, bpm, bpr, tab_text, n, key_str='', offset=0):
        self.tab_canvas.set_notes(tab_notes, bpm=bpm, bars_per_row=bpr,
                                   key=key_str, measure_offset=offset)
        self._set_tab_text(tab_text)
        self.btn_analyze.configure(state='normal')
        self.btn_export.configure(state='normal')
        self.status_lbl.configure(
            text=f'{n}개 음표 완료  {key_str}', text_color='#4CAF50')
        self.note_count_lbl.configure(text=f'{n}개 음표')
        # 현재 보기 모드에 맞게 표시
        self._on_view_change(self.view_var.get())

    def _on_error(self, err: str):
        self._set_tab_text(f'분석 오류:\n\n{err}')
        self.btn_analyze.configure(state='normal')
        self.status_lbl.configure(text='오류 발생', text_color='red')

    # ─────────────────────────── 보기 전환 ───────────────────────────

    def _on_view_change(self, view: str):
        if view == '악보':
            self._show_canvas(True)
            if self.tab_notes:
                try:
                    bpm = float(self.bpm_var.get())
                except ValueError:
                    bpm = 120.0
                bpr = int(self.bpr_var.get())
                self.tab_canvas.set_notes(self.tab_notes, bpm=bpm,
                                           bars_per_row=bpr,
                                           key=self.tab_canvas._key)
        else:
            self._show_canvas(False)
            if self.tab_notes:
                try:
                    bpm = float(self.bpm_var.get())
                except ValueError:
                    bpm = 120.0
                bpr = int(self.bpr_var.get())

                if view == '음표 목록':
                    text = self.renderer.note_list(self.tab_notes)
                else:
                    text = self.renderer.render_measures(
                        self.tab_notes, bpm=bpm, measures_per_row=bpr)
                self._set_tab_text(text)

    def _show_canvas(self, show: bool):
        if show:
            self.tab_text.grid_remove()
            self.tab_canvas.grid()
        else:
            self.tab_canvas.grid_remove()
            self.tab_text.grid()

    # ─────────────────────────── 내보내기 ────────────────────────────

    def export_tab(self):
        if not self.tab_notes:
            return
        path = filedialog.asksaveasfilename(
            title='TAB 저장',
            defaultextension='.txt',
            filetypes=[
                ('텍스트 파일', '*.txt'),
                ('PDF', '*.pdf'),
                ('PNG 이미지', '*.png'),
            ],
        )
        if not path:
            return
        if path.endswith('.pdf'):
            self._export_pdf(path)
        elif path.endswith('.png'):
            self._export_png(path)
        else:
            self._export_txt(path)

    def _export_txt(self, path: str):
        try:
            bpm = float(self.bpm_var.get())
        except ValueError:
            bpm = 120.0
        text = self.renderer.render_measures(
            self.tab_notes, bpm=bpm, measures_per_row=int(self.bpr_var.get()))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'베이스 TAB — {Path(self.current_file).name}\n')
            f.write(f'BPM: {bpm}  조율: {self.tuning_var.get()}\n\n')
            f.write(text)
        messagebox.showinfo('저장 완료', f'파일 저장됨:\n{path}')

    def _export_pdf(self, path: str):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import (SimpleDocTemplate, Preformatted,
                                             Paragraph)
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
        except ImportError:
            messagebox.showerror('오류', 'reportlab 미설치:\npip install reportlab')
            return

        try:
            bpm = float(self.bpm_var.get())
        except ValueError:
            bpm = 120.0
        text = self.renderer.render_measures(
            self.tab_notes, bpm=bpm,
            measures_per_row=int(self.bpr_var.get()))

        styles = getSampleStyleSheet()
        mono_style = ParagraphStyle(
            'Mono', parent=styles['Normal'],
            fontName='Courier', fontSize=7, leading=10)

        doc = SimpleDocTemplate(
            path, pagesize=A4,
            leftMargin=12 * mm, rightMargin=8 * mm,
            topMargin=18 * mm, bottomMargin=18 * mm)
        doc.build([
            Paragraph(f'Bass TAB: {Path(self.current_file).name}',
                      styles['Heading1']),
            Paragraph(f'BPM: {bpm}  |  조율: {self.tuning_var.get()}',
                      styles['Normal']),
            Preformatted(text, mono_style),
        ])
        messagebox.showinfo('저장 완료', f'PDF 저장됨:\n{path}')

    def _export_png(self, path: str):
        try:
            self.tab_canvas.export_image(path)
            messagebox.showinfo('저장 완료', f'이미지 저장됨:\n{path}')
        except Exception as exc:
            messagebox.showerror('오류', str(exc))

    # ─────────────────────────── 유틸 ────────────────────────────────

    def _set_tab_text(self, text: str):
        self.tab_text.configure(state='normal')
        self.tab_text.delete('1.0', 'end')
        self.tab_text.insert('end', text)
        self.tab_text.configure(state='disabled')

    def _ui(self, fn):
        self.after(0, fn)


# ──────────────────────────────────────────────────────────────────────────────

def run():
    app = BassTabApp()
    app.mainloop()
