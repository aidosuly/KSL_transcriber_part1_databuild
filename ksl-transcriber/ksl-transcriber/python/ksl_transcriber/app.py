"""Tkinter desktop GUI: camera feed, training, live recognition, transcript.

This is the Python-desktop counterpart to the browser version's js/app.js —
same behavior (train by recording bursts, recognize with a stability-gated
auto-commit, build a Kazakh transcript), driven by OpenCV + MediaPipe
instead of the browser's camera APIs.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import hand_tracking as ht
from dataset import Dataset
from labels import CAT_ORDER, LabelStore

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - surfaced in the UI instead
    mp = None

STABLE_FRAMES = 10
RECORD_DURATION_S = 1.5
FRAME_INTERVAL_MS = 20  # ~50fps polling; actual camera fps may be lower


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("KSL Live Transcriber (Python)")
        self.root.geometry("1180x760")

        self.labels = LabelStore()
        self.dataset = Dataset()

        self.train_category = 'alphabet'
        self.recog_category = 'alphabet'
        self.train_selected_label: str | None = None

        self.cap = None
        self.hands = None
        self.running = False
        self.latest_result = None  # (vector, hand_count, raw_multi_hand_landmarks)

        self.recognition_on = tk.BooleanVar(value=False)
        self.auto_commit = tk.BooleanVar(value=False)
        self.history: list = []
        self.last_committed_label = None
        self.hand_cleared_since_commit = True
        self.current_prediction = None

        self.recording = False
        self.record_label = None
        self.record_end_time = 0.0
        self.record_samples: list = []

        self.buffer = ''

        self._build_ui()
        self._refresh_label_grid()
        self._refresh_data_tab()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = self.root
        main = ttk.Frame(root, padding=10)
        main.pack(fill='both', expand=True)

        left = ttk.Frame(main)
        left.pack(side='left', fill='both', expand=True, padx=(0, 10))
        right = ttk.Frame(main, width=420)
        right.pack(side='left', fill='y')

        # --- camera panel ---
        cam_box = ttk.LabelFrame(left, text="Camera", padding=8)
        cam_box.pack(fill='both', expand=True)
        self.video_label = ttk.Label(cam_box, background='black')
        self.video_label.pack(fill='both', expand=True)

        cam_controls = ttk.Frame(cam_box)
        cam_controls.pack(fill='x', pady=(8, 0))
        self.btn_start = ttk.Button(cam_controls, text="Start camera", command=self.start_camera)
        self.btn_start.pack(side='left')
        self.btn_stop = ttk.Button(cam_controls, text="Stop camera", command=self.stop_camera, state='disabled')
        self.btn_stop.pack(side='left', padx=(6, 0))
        self.hand_badge = ttk.Label(cam_controls, text="Hands: 0")
        self.hand_badge.pack(side='left', padx=12)

        status = ttk.Frame(cam_box)
        status.pack(fill='x', pady=(6, 0))
        self.mode_label_var = tk.StringVar(value='Alphabet')
        self.recog_label_var = tk.StringVar(value='off')
        ttk.Label(status, textvariable=tk.StringVar(value='Mode:')).pack(side='left')
        ttk.Label(status, textvariable=self.mode_label_var, font=('', 10, 'bold')).pack(side='left', padx=(4, 16))
        ttk.Label(status, text='Recognition:').pack(side='left')
        ttk.Label(status, textvariable=self.recog_label_var, font=('', 10, 'bold')).pack(side='left', padx=(4, 0))

        pred_box = ttk.Frame(cam_box, padding=8, relief='groove')
        pred_box.pack(fill='x', pady=(8, 0))
        self.pred_symbol_var = tk.StringVar(value='—')
        ttk.Label(pred_box, textvariable=self.pred_symbol_var, font=('Consolas', 26, 'bold')).pack(side='left', padx=(0, 12))
        pred_meta = ttk.Frame(pred_box)
        pred_meta.pack(side='left', fill='x', expand=True)
        self.pred_text_var = tk.StringVar(value='Recognition is off')
        ttk.Label(pred_meta, textvariable=self.pred_text_var).pack(anchor='w')
        self.pred_conf = ttk.Progressbar(pred_meta, maximum=100, value=0)
        self.pred_conf.pack(fill='x', pady=(4, 0))
        ttk.Button(pred_box, text="+ Add", command=self._add_current_prediction).pack(side='left', padx=(12, 0))

        ttk.Label(cam_box, text="Space = add current sign · Enter = finish word · Backspace = delete",
                  foreground='#666').pack(anchor='w', pady=(6, 0))

        # --- tabs ---
        notebook = ttk.Notebook(right)
        notebook.pack(fill='both', expand=True)

        self.tab_train = ttk.Frame(notebook, padding=8)
        self.tab_recognize = ttk.Frame(notebook, padding=8)
        self.tab_data = ttk.Frame(notebook, padding=8)
        notebook.add(self.tab_train, text='Train')
        notebook.add(self.tab_recognize, text='Recognize')
        notebook.add(self.tab_data, text='Data')

        self._build_train_tab()
        self._build_recognize_tab()
        self._build_data_tab()

        # --- transcript panel ---
        transcript_box = ttk.LabelFrame(root, text="Transcript", padding=10)
        transcript_box.pack(fill='x', padx=10, pady=(0, 10))
        self.buffer_var = tk.StringVar(value='Buffer: ')
        ttk.Label(transcript_box, textvariable=self.buffer_var, font=('Consolas', 13), foreground='#b8860b').pack(anchor='w')
        self.transcript_text = tk.Text(transcript_box, height=4, font=('Consolas', 12), wrap='word')
        self.transcript_text.pack(fill='x', pady=(4, 6))
        t_controls = ttk.Frame(transcript_box)
        t_controls.pack(fill='x')
        ttk.Button(t_controls, text="Space", command=lambda: self.transcript_text.insert('end', ' ')).pack(side='left')
        ttk.Button(t_controls, text="⏎ Finish word", command=self._finish_word).pack(side='left', padx=6)
        ttk.Button(t_controls, text="⌫ Backspace", command=self._backspace).pack(side='left')
        ttk.Button(t_controls, text="Clear", command=self._clear_all).pack(side='left', padx=6)
        ttk.Button(t_controls, text="Copy", command=self._copy_transcript).pack(side='left')

        root.bind('<space>', self._on_key_space)
        root.bind('<Return>', lambda e: self._finish_word())
        root.bind('<BackSpace>', self._on_key_backspace)

    def _build_train_tab(self):
        t = self.tab_train
        self.cat_row_train = ttk.Frame(t)
        self.cat_row_train.pack(fill='x')

        self.add_word_panel = ttk.Frame(t)
        self.new_word_kk = tk.StringVar()
        self.new_word_gloss = tk.StringVar()
        ttk.Entry(self.add_word_panel, textvariable=self.new_word_kk, width=16).pack(side='left', padx=(0, 4))
        ttk.Entry(self.add_word_panel, textvariable=self.new_word_gloss, width=16).pack(side='left', padx=(0, 4))
        ttk.Button(self.add_word_panel, text="+ Add word", command=self._submit_new_word).pack(side='left')
        # shown only when the Words category is selected (see _select_train_category)

        grid_frame = ttk.Frame(t)
        grid_frame.pack(fill='both', expand=True, pady=(8, 8))
        canvas = tk.Canvas(grid_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(grid_frame, orient='vertical', command=canvas.yview)
        self.label_grid_inner = ttk.Frame(canvas)
        self.label_grid_inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.label_grid_inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set, height=280)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        train_controls = ttk.Frame(t)
        train_controls.pack(fill='x')
        self.train_cur_var = tk.StringVar(value='—')
        self.train_gloss_var = tk.StringVar(value='Pick a label above to train it.')
        ttk.Label(train_controls, textvariable=self.train_cur_var, font=('Consolas', 16, 'bold')).pack(side='left')
        ttk.Label(train_controls, textvariable=self.train_gloss_var, foreground='#666').pack(side='left', padx=8, fill='x', expand=True)
        self.btn_record = ttk.Button(train_controls, text="● Record (1.5s)", command=self._start_recording, state='disabled')
        self.btn_record.pack(side='right')

        self.record_progress = ttk.Progressbar(t, maximum=100, value=0)
        self.record_progress.pack(fill='x', pady=(6, 0))

        ttk.Label(t, text="Hold the sign steady in frame, then click Record. Repeat 5-10x per label "
                          "from slightly different angles for better accuracy.",
                  foreground='#666', wraplength=380, justify='left').pack(anchor='w', pady=(8, 0))

        self._build_cat_row(self.cat_row_train, lambda: self.train_category, self._select_train_category)

    def _build_recognize_tab(self):
        t = self.tab_recognize
        self.cat_row_recog = ttk.Frame(t)
        self.cat_row_recog.pack(fill='x')
        self._build_cat_row(self.cat_row_recog, lambda: self.recog_category, self._select_recog_category)

        ttk.Checkbutton(t, text="Recognition running", variable=self.recognition_on,
                         command=self._on_toggle_recognition).pack(anchor='w', pady=(12, 4))
        ttk.Checkbutton(t, text="Auto-commit stable signs (else use Space / + Add)",
                         variable=self.auto_commit).pack(anchor='w')
        ttk.Label(t, text="Recognition only compares against labels in the selected category — "
                          "narrower category, more accurate matches, especially with few samples.",
                  foreground='#666', wraplength=380, justify='left').pack(anchor='w', pady=(12, 0))

    def _build_data_tab(self):
        t = self.tab_data
        stat_row = ttk.Frame(t)
        stat_row.pack(fill='x', pady=(0, 8))
        self.stat_total_var = tk.StringVar(value='0')
        self.stat_labels_var = tk.StringVar(value='0')
        self.stat_coverage_var = tk.StringVar(value='0%')
        for label, var in (('Samples', self.stat_total_var), ('Labels with data', self.stat_labels_var),
                           ('Alphabet+digits covered', self.stat_coverage_var)):
            box = ttk.Frame(stat_row, relief='groove', padding=6)
            box.pack(side='left', fill='x', expand=True, padx=(0, 6))
            ttk.Label(box, textvariable=var, font=('Consolas', 14, 'bold')).pack()
            ttk.Label(box, text=label, foreground='#666', font=('', 8)).pack()

        columns = ('label', 'category', 'samples')
        self.data_tree = ttk.Treeview(t, columns=columns, show='headings', height=10)
        for col, text in zip(columns, ('Label', 'Category', 'Samples')):
            self.data_tree.heading(col, text=text)
        self.data_tree.column('label', width=180)
        self.data_tree.column('category', width=90)
        self.data_tree.column('samples', width=70, anchor='center')
        self.data_tree.pack(fill='both', expand=True, pady=(0, 8))

        btns = ttk.Frame(t)
        btns.pack(fill='x')
        ttk.Button(btns, text="Export dataset", command=self._export_dataset).pack(side='left')
        ttk.Button(btns, text="Import dataset", command=self._import_dataset).pack(side='left', padx=6)
        ttk.Button(btns, text="Reset all data", command=self._reset_all_data).pack(side='left')

        ttk.Label(t, text="Data auto-saves to python/data/ on disk. Export regularly — it's the same "
                          "labeled dataset format the browser version and Phase 2 training will use.",
                  foreground='#666', wraplength=380, justify='left').pack(anchor='w', pady=(8, 0))

    def _build_cat_row(self, container, get_cat, on_select):
        for child in container.winfo_children():
            child.destroy()
        for key in CAT_ORDER:
            style = 'Accent.TButton' if get_cat() == key else 'TButton'
            b = ttk.Button(container, text=self.labels.category_name(key), style=style,
                           command=lambda k=key: on_select(k))
            b.pack(side='left', padx=(0, 4))

    # ------------------------------------------------------------------
    # Category selection
    # ------------------------------------------------------------------
    def _select_train_category(self, key):
        self.train_category = key
        self.train_selected_label = None
        self.btn_record.configure(state='disabled')
        self.train_cur_var.set('—')
        self.train_gloss_var.set('Pick a label above to train it.')
        if key == 'words':
            self.add_word_panel.pack(fill='x', pady=(6, 4))
        else:
            self.add_word_panel.pack_forget()
        self._build_cat_row(self.cat_row_train, lambda: self.train_category, self._select_train_category)
        self._refresh_label_grid()
        self._refresh_data_tab()

    def _select_recog_category(self, key):
        self.recog_category = key
        self.mode_label_var.set(self.labels.category_name(key))
        self.history = []
        self.last_committed_label = None
        self._build_cat_row(self.cat_row_recog, lambda: self.recog_category, self._select_recog_category)

    # ------------------------------------------------------------------
    # Label grid (Train tab)
    # ------------------------------------------------------------------
    def _refresh_label_grid(self):
        for child in self.label_grid_inner.winfo_children():
            child.destroy()
        items = self.labels.items(self.train_category)
        cols = 6
        for idx, item in enumerate(items):
            count = self.dataset.count_for(item.label)
            text = f"{item.label}\n{item.gloss[:10]}\n({count})"
            style = 'Selected.TButton' if item.label == self.train_selected_label else 'TButton'
            cell = ttk.Frame(self.label_grid_inner)
            cell.grid(row=idx // cols, column=idx % cols, padx=2, pady=2, sticky='nsew')
            btn = ttk.Button(cell, text=text, style=style,
                              command=lambda lb=item.label: self._select_train_label(lb))
            btn.pack(fill='both', expand=True)
            if item.custom:
                del_btn = tk.Button(cell, text='×', fg='white', bg='#c1402c', bd=0,
                                     command=lambda lb=item.label: self._remove_custom_word(lb))
                del_btn.place(relx=1.0, y=0, anchor='ne', width=16, height=16)

    def _select_train_label(self, label):
        self.train_selected_label = label
        item = self.labels.find(self.train_category, label)
        self.train_cur_var.set(label)
        count = self.dataset.count_for(label)
        self.train_gloss_var.set(f"{item.gloss} — samples recorded: {count}")
        self.btn_record.configure(state='normal')
        self._refresh_label_grid()

    def _remove_custom_word(self, label):
        if not messagebox.askyesno("Remove word", f'Remove "{label}" and its recorded samples?'):
            return
        self.labels.remove_custom_word(label)
        self.dataset.remove_label(label)
        if self.train_selected_label == label:
            self.train_selected_label = None
            self.btn_record.configure(state='disabled')
            self.train_cur_var.set('—')
            self.train_gloss_var.set('Pick a label above to train it.')
        self._refresh_label_grid()
        self._refresh_data_tab()

    def _submit_new_word(self):
        ok, error = self.labels.add_word(self.new_word_kk.get(), self.new_word_gloss.get())
        if ok:
            self.new_word_kk.set('')
            self.new_word_gloss.set('')
            self._refresh_label_grid()
            self._refresh_data_tab()
        else:
            messagebox.showinfo("Add word", error)

    # ------------------------------------------------------------------
    # Camera + MediaPipe
    # ------------------------------------------------------------------
    def start_camera(self):
        if mp is None:
            messagebox.showerror("MediaPipe missing",
                                  "mediapipe is not installed. Run: pip install -r requirements.txt")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open the webcam (device 0).")
            self.cap = None
            return
        if self.hands is None:
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=2, model_complexity=1,
                min_detection_confidence=0.6, min_tracking_confidence=0.5,
            )
        self.running = True
        self.btn_start.configure(state='disabled')
        self.btn_stop.configure(state='normal')
        self._update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image='')
        self.hand_badge.configure(text='Hands: 0')
        self.btn_start.configure(state='normal')
        self.btn_stop.configure(state='disabled')

    def _update_frame(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)  # mirror, like the browser version
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            self._on_results(rgb, results)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # keep a reference or Tkinter garbage-collects it
            self.video_label.configure(image=imgtk)
        self.root.after(FRAME_INTERVAL_MS, self._update_frame)

    def _on_results(self, rgb_frame, results):
        multi_lms = results.multi_hand_landmarks  # list of landmark_list objects, or None
        multi_handedness = results.multi_handedness
        n = len(multi_lms) if multi_lms else 0
        self.hand_badge.configure(text=f'Hands: {n}')
        self._draw_skeleton(rgb_frame, multi_lms)

        # Adapt MediaPipe's result shape to hand_tracking's plain-data API
        # (a list of landmark lists, a parallel list of handedness labels)
        # so hand_tracking.py never has to know about MediaPipe's object types.
        landmark_lists = [hand_lms.landmark for hand_lms in multi_lms] if multi_lms else []
        labels = [h.classification[0].label for h in multi_handedness] if multi_handedness else []
        vector, hand_count = ht.build_vector(landmark_lists, labels)
        self.latest_result = (vector, hand_count)

        if self.recording:
            self._record_tick(vector, hand_count)

        if self.recognition_on.get():
            candidates = [i.label for i in self.labels.items(self.recog_category)]
            result = ht.classify(vector, hand_count, candidates, self.dataset.samples)
            self.current_prediction = result
            self._update_prediction_ui(result)
            self._push_history(result['label'] if result else None)
            if self.auto_commit.get():
                self._maybe_auto_commit()
        else:
            self.current_prediction = None
            self._update_prediction_ui(None)

    def _draw_skeleton(self, rgb_frame, multi_lms):
        if not multi_lms:
            return
        h, w = rgb_frame.shape[:2]
        for hand_lms in multi_lms:
            pts = [(int(p.x * w), int(p.y * h)) for p in hand_lms.landmark]
            for a, b in ht.HAND_CONNECTIONS:
                cv2.line(rgb_frame, pts[a], pts[b], (63, 188, 232), 2)
            for x, y in pts:
                cv2.circle(rgb_frame, (x, y), 3, (217, 154, 43), -1)

    # ------------------------------------------------------------------
    # Recording bursts
    # ------------------------------------------------------------------
    def _start_recording(self):
        if not self.train_selected_label:
            return
        if not self.running:
            messagebox.showinfo("Start the camera", "Start the camera first.")
            return
        self.recording = True
        self.record_label = self.train_selected_label
        self.record_end_time = time.time() + RECORD_DURATION_S
        self.record_samples = []
        self.btn_record.configure(state='disabled', text='● Recording…')

    def _record_tick(self, vector, hand_count):
        if vector and hand_count > 0:
            self.record_samples.append((hand_count, vector))
        elapsed = RECORD_DURATION_S - max(0.0, self.record_end_time - time.time())
        pct = min(100, int((elapsed / RECORD_DURATION_S) * 100))
        self.record_progress.configure(value=pct)
        if time.time() >= self.record_end_time:
            self._finish_recording()

    def _finish_recording(self):
        self.recording = False
        self.record_progress.configure(value=0)
        self.btn_record.configure(state='normal', text='● Record (1.5s)')
        label = self.record_label
        by_hand_count = {}
        for hc, vec in self.record_samples:
            by_hand_count.setdefault(hc, []).append(vec)
        added = 0
        for hc, vecs in by_hand_count.items():
            added += self.dataset.add_samples(label, self.train_category, hc, vecs)
        item = self.labels.find(self.train_category, label)
        self.train_gloss_var.set(f"{item.gloss if item else ''} — samples recorded: {self.dataset.count_for(label)}")
        self._refresh_label_grid()
        self._refresh_data_tab()
        if added == 0:
            messagebox.showinfo("No samples", "No hand detected during recording — try again.")

    # ------------------------------------------------------------------
    # Prediction / transcript
    # ------------------------------------------------------------------
    def _update_prediction_ui(self, result):
        if not result:
            self.pred_symbol_var.set('—')
            self.pred_text_var.set('No confident match' if self.recognition_on.get() else 'Recognition is off')
            self.pred_conf.configure(value=0)
            self.recog_label_var.set('on' if self.recognition_on.get() else 'off')
            return
        self.pred_symbol_var.set(result['label'])
        item = self.labels.find(self.recog_category, result['label'])
        gloss = f"{item.gloss} · " if item else ''
        self.pred_text_var.set(f"{gloss}{round(result['confidence'] * 100)}% confidence")
        self.pred_conf.configure(value=round(result['confidence'] * 100))
        self.recog_label_var.set('on')

    def _push_history(self, label):
        self.history.append(label)
        if len(self.history) > 20:
            self.history.pop(0)
        if label is None:
            self.hand_cleared_since_commit = True

    def _maybe_auto_commit(self):
        if len(self.history) < STABLE_FRAMES:
            return
        recent = self.history[-STABLE_FRAMES:]
        first = recent[0]
        if first is None or any(l != first for l in recent):
            return
        if first == self.last_committed_label and not self.hand_cleared_since_commit:
            return
        self._commit_label(first)
        self.last_committed_label = first
        self.hand_cleared_since_commit = False

    def _commit_label(self, label):
        item = self.labels.find(self.recog_category, label)
        buffered = item.buffered if item else True
        if buffered:
            self.buffer += label
            self.buffer_var.set(f'Buffer: {self.buffer}')
        else:
            current = self.transcript_text.get('1.0', 'end-1c')
            sep = ' ' if current and not current.endswith(' ') else ''
            self.transcript_text.insert('end', f'{sep}{label} ')

    def _add_current_prediction(self):
        if self.current_prediction:
            self._commit_label(self.current_prediction['label'])

    def _finish_word(self):
        if not self.buffer:
            return
        current = self.transcript_text.get('1.0', 'end-1c')
        sep = ' ' if current and not current.endswith(' ') else ''
        self.transcript_text.insert('end', f'{sep}{self.buffer} ')
        self.buffer = ''
        self.buffer_var.set('Buffer: ')

    def _backspace(self):
        if self.buffer:
            self.buffer = self.buffer[:-1]
            self.buffer_var.set(f'Buffer: {self.buffer}')
        else:
            content = self.transcript_text.get('1.0', 'end-1c')
            self.transcript_text.delete('1.0', 'end')
            self.transcript_text.insert('1.0', content[:-1])

    def _clear_all(self):
        self.buffer = ''
        self.buffer_var.set('Buffer: ')
        self.transcript_text.delete('1.0', 'end')

    def _copy_transcript(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.transcript_text.get('1.0', 'end-1c'))

    def _on_key_space(self, event):
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        self._add_current_prediction()
        return 'break'

    def _on_key_backspace(self, event):
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        self._backspace()
        return 'break'

    def _on_toggle_recognition(self):
        self.history = []
        self.last_committed_label = None
        if not self.recognition_on.get():
            self._update_prediction_ui(None)

    # ------------------------------------------------------------------
    # Data tab
    # ------------------------------------------------------------------
    def _refresh_data_tab(self):
        for row in self.data_tree.get_children():
            self.data_tree.delete(row)
        total = 0
        labels_with_data = 0
        covered_core = 0
        core_labels = set(
            [i.label for i in self.labels.items('alphabet')] +
            [i.label for i in self.labels.items('numbers')]
        )
        for cat_key in CAT_ORDER:
            for item in self.labels.items(cat_key):
                c = self.dataset.count_for(item.label)
                total += c
                if c > 0:
                    labels_with_data += 1
                    if item.label in core_labels:
                        covered_core += 1
                self.data_tree.insert('', 'end', values=(f"{item.label} ({item.gloss})",
                                                          self.labels.category_name(cat_key), c))
        self.stat_total_var.set(str(total))
        self.stat_labels_var.set(str(labels_with_data))
        pct = round((covered_core / len(core_labels)) * 100) if core_labels else 0
        self.stat_coverage_var.set(f'{pct}%')

    def _export_dataset(self):
        path = filedialog.asksaveasfilename(defaultextension='.json',
                                             initialfile='ksl_dataset.json',
                                             filetypes=[('JSON', '*.json')])
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.dataset.export_json())
        messagebox.showinfo("Exported", f"Dataset exported to:\n{path}")

    def _import_dataset(self):
        path = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if not path:
            return
        merge = messagebox.askyesno("Import", "Merge with your current data?\n(No = replace entirely)")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.dataset.import_json(f.read(), merge)
            self._refresh_label_grid()
            self._refresh_data_tab()
        except (ValueError, OSError) as e:
            messagebox.showerror("Import failed", str(e))

    def _reset_all_data(self):
        if messagebox.askyesno("Reset all data",
                                "Delete ALL recorded samples? Custom words stay.\n"
                                "This cannot be undone (export first if unsure)."):
            self.dataset.reset_all()
            self._refresh_label_grid()
            self._refresh_data_tab()

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass
    style.configure('Selected.TButton', foreground='#0090c7')
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
