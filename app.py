import sys
import numpy as np
import sounddevice as sd
import threading
import struct
import random
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QLabel, QComboBox, QCheckBox, QGroupBox, 
    QStatusBar, QTabWidget, QFileDialog, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QPainter, QImage, QColor, QPen, QLinearGradient, QBrush, QPolygonF

# --- Configuration ---
GRID_W = 64       # X-axis: Time
GRID_H = 32       # Y-axis: Pitch
SAMPLE_RATE = 44100

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note_name(freq):
    if freq <= 0: return ""
    midi = int(round(12 * np.log2(freq / 440.0) + 69))
    octave = (midi // 12) - 1
    name = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"

def freq_to_midi(freq):
    return int(round(12 * np.log2(freq / 440.0) + 69))

# Built-in standard rules
RULES = {
    "HighLife": ((3, 6), (2, 3), 2),
    "Conway": ((3,), (2, 3), 2),
    "Seeds": ((2,), (), 2),
    "Maze": ((3,), (1, 2, 3, 4, 5), 2),
    "Replicator": ((1, 3, 5, 7), (1, 3, 5, 7), 2),
    "Day & Night": ((3, 6, 7, 8), (3, 4, 6, 7, 8), 2),
    "Gnarl": ((1,), (1,), 2),
    "Assimilation": ((3, 4, 5), (4, 5, 6, 7), 2),
    "Coral": ((3,), (4, 5, 6, 7, 8), 2),
    "Star Wars (Gen)": ((2,), (3,4,5,6,7,8), 3),
    "Bloom (Gen)": ((3,), (4,5,6,7,8), 3),
    "Brians Brain (Gen)": ((2,), (), 3),
    "Burst (Gen)": ((2,), (4,5,6,7,8), 4),
}

PATTERNS = {
    "Glider": [(0,1), (1,2), (2,0), (2,1), (2,2)],
    "LWSS": [(0,1), (0,4), (1,0), (2,0), (2,4), (3,0), (3,1), (3,2), (3,3)],
    "Acorn": [(0,1), (1,3), (2,0), (2,1), (2,4), (2,5), (2,6)],
    "R-Pentomino": [(0,1), (0,2), (1,0), (1,1), (2,1)],
    "Pulsar": [
        (0,2),(0,3),(0,4),(0,8),(0,9),(0,10),
        (2,0),(2,5),(2,7),(2,12),(3,0),(3,5),(3,7),(3,12),(4,0),(4,5),(4,7),(4,12),
        (5,2),(5,3),(5,4),(5,8),(5,9),(5,10),
        (7,2),(7,3),(7,4),(7,8),(7,9),(7,10),
        (8,0),(8,5),(8,7),(8,12),(9,0),(9,5),(9,7),(9,12),(10,0),(10,5),(10,7),(10,12),
        (12,2),(12,3),(12,4),(12,8),(12,9),(12,10)
    ],
    "Pentadecathlon": [(0,1),(1,1),(2,0),(2,2),(3,1),(4,1),(5,1),(6,1),(7,0),(7,2),(8,1),(9,1)],
    "Gosper Glider Gun": [
        (0,24),(1,22),(1,24),(2,12),(2,13),(2,20),(2,21),(2,34),(2,35),
        (3,11),(3,15),(3,20),(3,21),(3,34),(3,35),(4,0),(4,1),(4,10),(4,16),(4,20),(4,21),
        (5,0),(5,1),(5,10),(5,14),(5,16),(5,17),(5,20),(5,21),(6,10),(6,16),(6,17),
        (7,11),(7,15),(8,12),(8,13)
    ],
    "Blinker": [(0,0), (0,1), (0,2)],
    "Toad": [(0,1), (0,2), (0,3), (1,0), (1,1), (1,2)],
    # Wireworld Logic Gates (Conductor=0.5, Head=1.0, Tail=0.2)
    "WW Clock (Fast)": [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1.0),(1,7)],
    "WW XOR Gate": [
        (0,2),(0,3),(0,6),(0,7),(1,2),(1,3),(1,6),(1,7),
        (2,3),(2,6),(3,3),(3,4),(3,5),(3,6),(4,4),(4,5),
        (5,4),(5,5),(6,3),(6,4),(6,5),(6,6),(7,3),(7,6),
        (8,2),(8,3),(8,6),(8,7),(9,2),(9,3),(9,6),(9,7)
    ]
}

# --- 1. Cellular Automata Engine ---
class CAEngine:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.rule = "HighLife"
        self.rule_type = "Life"
        self.custom_birth = (3,)
        self.custom_survive = (2, 3)
        self.gen_states = 2
        self.mutation_rate = 0.0
        self.noise_time = 0.0
        self.grid = np.random.choice([0.0, 1.0], size=(height, width), p=[0.9, 0.1]).astype(np.float32)
        self.ant_x = width // 2
        self.ant_y = height // 2
        self.ant_dir = 0
        self._neighbors = np.zeros((height, width), dtype=np.int8)

    def set_custom_rule(self, rule_str):
        # Parses standard Bxx/Syy/Cz notation (e.g. B3/S23/2, B2/S/3)
        try:
            parts = rule_str.upper().split('/')
            b_part = next((p for p in parts if p.startswith('B')), 'B3')
            s_part = next((p for p in parts if p.startswith('S')), 'S23')
            c_part = next((p for p in parts if p.startswith('C')), 'C2')
            
            self.custom_birth = tuple(int(c) for c in b_part[1:])
            self.custom_survive = tuple(int(c) for c in s_part[1:])
            self.gen_states = max(2, int(c_part[1:]))
            
            if self.gen_states > 2:
                self.rule_type = "Generations"
            else:
                self.rule_type = "Life"
            self.rule = "Custom"
            return True
        except:
            return False

    def set_builtin_rule(self, rule_name):
        self.rule = rule_name
        if rule_name == "Procedural Noise": 
            self.rule_type = "Noise"
        elif rule_name == "Langton's Ant": 
            self.rule_type = "Ant"
        elif rule_name == "Wireworld": 
            self.rule_type = "Wireworld"
        elif "(Gen)" in rule_name:
            self.rule_type = "Generations"
            b, s, c = RULES[rule_name]
            self.custom_birth, self.custom_survive, self.gen_states = b, s, c
        else:
            self.rule_type = "Life"
            b, s, c = RULES[rule_name]
            self.custom_birth, self.custom_survive, self.gen_states = b, s, c

    def step(self):
        if self.rule_type == "Noise": self.step_noise()
        elif self.rule_type == "Ant": self.step_langtons_ant()
        elif self.rule_type == "Wireworld": self.step_wireworld()
        elif self.rule_type == "Generations": self.step_generations()
        else: self.step_2state()

    def step_2state(self):
        alive = self.grid > 0.95 # Only fully alive cells count for standard life
        self._neighbors.fill(0)
        neighbors = self._neighbors
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1: continue
                neighbors += np.roll(np.roll(alive, i-1, axis=0), j-1, axis=1)
        
        birth_rules, survive_rules, _ = RULES.get(self.rule, (self.custom_birth, self.custom_survive, 2))
            
        birth = np.zeros_like(self.grid, dtype=bool)
        for b in birth_rules: birth |= (neighbors == b)
        birth &= ~alive
        
        survive = np.zeros_like(self.grid, dtype=bool)
        for s in survive_rules: survive |= (neighbors == s)
        survive &= alive
        
        self.grid *= 0.85 
        self.grid[birth | survive] = 1.0
        
        if self.mutation_rate > 0:
            mutations = np.random.rand(self.height, self.width) < self.mutation_rate
            self.grid[mutations] = 1.0 - self.grid[mutations]
            
        if np.sum(alive) < 5 and self.mutation_rate == 0:
            self.grid[np.random.randint(0, self.height), np.random.randint(0, self.width)] = 1.0

    def step_generations(self):
        # Multi-state decay rules. State 1.0 = Alive. States 0.8 -> 0.2 = Dying.
        alive = self.grid > 0.95
        self._neighbors.fill(0)
        neighbors = self._neighbors
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1: continue
                neighbors += np.roll(np.roll(alive, i-1, axis=0), j-1, axis=1)
                
        birth_rules = self.custom_birth
        survive_rules = self.custom_survive
        C = self.gen_states
        
        new_grid = np.zeros_like(self.grid)
        
        # Survive
        survive = np.zeros_like(self.grid, dtype=bool)
        for s in survive_rules: survive |= (neighbors == s)
        survive &= alive
        new_grid[survive] = 1.0
        
        # Birth
        birth = np.zeros_like(self.grid, dtype=bool)
        for b in birth_rules: birth |= (neighbors == b)
        birth &= ~alive
        new_grid[birth] = 1.0
        
        # Decay
        for i in range(2, C):
            val = 1.0 - ((i-1) * (0.8 / (C-2)))
            new_grid[self.grid == val] = val - (0.8 / (C-2))
            
        self.grid = new_grid
        
        if self.mutation_rate > 0:
            mutations = np.random.rand(self.height, self.width) < self.mutation_rate
            self.grid[mutations] = 1.0 # Mutations spark new life
            
        if np.sum(alive) < 2 and self.mutation_rate == 0:
            self.grid[np.random.randint(0, self.height), np.random.randint(0, self.width)] = 1.0

    def step_wireworld(self):
        # 1.0 = Head, 0.2 = Tail, 0.5 = Conductor, 0.0 = Empty
        heads = self.grid > 0.95
        self._neighbors.fill(0)
        neighbors = self._neighbors
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1: continue
                neighbors += np.roll(np.roll(heads, i-1, axis=0), j-1, axis=1)
                
        new_grid = np.copy(self.grid)
        new_grid[self.grid > 0.95] = 0.2       # Head -> Tail
        new_grid[np.isclose(self.grid, 0.2)] = 0.5 # Tail -> Conductor
        # Conductor -> Head if 1 or 2 neighbors
        new_grid[(np.isclose(self.grid, 0.5)) & ((neighbors == 1) | (neighbors == 2))] = 1.0
        
        self.grid = new_grid

    def step_noise(self):
        self.noise_time += 0.1
        y, x = np.meshgrid(np.linspace(0, 4, self.height), np.linspace(0, 8, self.width), indexing='ij')
        n = np.sin(x + self.noise_time) + np.sin(y * 0.5 + self.noise_time * 1.3) + np.sin((x + y) * 0.7 + self.noise_time * 0.8)
        self.grid = (n / 3.0) * 0.5 + 0.5

    def step_langtons_ant(self):
        if self.grid[self.ant_y, self.ant_x] > 0.5:
            self.ant_dir = (self.ant_dir + 1) % 4
            self.grid[self.ant_y, self.ant_x] = 0.0
        else:
            self.ant_dir = (self.ant_dir - 1) % 4
            self.grid[self.ant_y, self.ant_x] = 1.0
        if self.ant_dir == 0: self.ant_y = (self.ant_y - 1) % self.height
        elif self.ant_dir == 1: self.ant_x = (self.ant_x + 1) % self.width
        elif self.ant_dir == 2: self.ant_y = (self.ant_y + 1) % self.height
        elif self.ant_dir == 3: self.ant_x = (self.ant_x - 1) % self.width


# --- 2. Effects (Reverb & Chorus) ---
class Reverb:
    def __init__(self, sr):
        self.sr = sr
        self.combs = []
        delays = [0.0297, 0.0371, 0.0411, 0.0501]
        decays = [0.85, 0.82, 0.80, 0.78]
        for d, dec in zip(delays, decays):
            self.combs.append({
                'buf': np.zeros(int(sr * d)), 'ptr': 0, 'decay': dec, 'lp': 0.0
            })
            
    def process_block(self, left, right):
        out_l = np.zeros_like(left)
        out_r = np.zeros_like(right)
        for i in range(len(left)):
            in_mono = (left[i] + right[i]) * 0.5
            rev = 0.0
            for c in self.combs:
                val = c['buf'][c['ptr']]
                c['lp'] = (val * 0.5) + (c['lp'] * 0.5) 
                c['buf'][c['ptr']] = in_mono + c['lp'] * c['decay']
                c['ptr'] = (c['ptr'] + 1) % len(c['buf'])
                rev += val
            out_l[i] = rev * 0.2
            out_r[i] = rev * 0.2
        return out_l, out_r

class Chorus:
    def __init__(self, sr):
        self.sr = sr
        self.buf = np.zeros(int(sr * 0.05)) 
        self.ptr = 0
        self.phase = 0.0
        
    def process_block(self, left, right, depth):
        out_l = np.zeros_like(left)
        out_r = np.zeros_like(right)
        buf_len = len(self.buf)
        
        inc = 2.0 * np.pi * 1.5 / self.sr
        lfo_l = np.sin(np.arange(len(left)) * inc + self.phase)
        lfo_r = np.sin(np.arange(len(right)) * inc + self.phase + np.pi/2)
        self.phase += inc * len(left)
        
        delay_l = 0.015 * self.sr + lfo_l * 0.005 * self.sr
        delay_r = 0.015 * self.sr + lfo_r * 0.005 * self.sr
        
        read_ptr_l = (self.ptr - delay_l + buf_len) % buf_len
        read_ptr_r = (self.ptr - delay_r + buf_len) % buf_len
        
        base_l = np.floor(read_ptr_l).astype(int)
        frac_l = read_ptr_l - base_l
        next_l = (base_l + 1) % buf_len
        out_l = self.buf[base_l] * (1-frac_l) + self.buf[next_l] * frac_l
        
        base_r = np.floor(read_ptr_r).astype(int)
        frac_r = read_ptr_r - base_r
        next_r = (base_r + 1) % buf_len
        out_r = self.buf[base_r] * (1-frac_r) + self.buf[next_r] * frac_r
        
        write_indices = (self.ptr + np.arange(len(left))) % buf_len
        self.buf[write_indices] = (left + right) * 0.5
        self.ptr = (self.ptr + len(left)) % buf_len
        
        return (left * (1.0 - depth)) + (out_l * depth), (right * (1.0 - depth)) + (out_r * depth)

# --- 3. Audio Engine ---
class AudioEngine:
    def __init__(self, sample_rate=44100, grid_shape=(32, 64)):
        self.sample_rate = sample_rate
        self.h, self.w = grid_shape
        self.grid = np.zeros(grid_shape, dtype=np.float32)
        self.processed_grid = np.zeros(grid_shape, dtype=np.float32)
        
        self.waveform = 'sine'
        self.scale = 'pentatonic'
        self.volume = 0.8
        self.octave_shift = 0
        self.sweep_duration = 4.0
        self.spectral_blur = 0.0
        
        self.unison = 0.1
        self.fm_amount = 2.0
        self.drive = 1.0
        self.pluck_amt = 0.5
        self.attack = 0.01
        self.release = 0.1
        self.wt_pos = 0.0
        self.sub_octave = -1
        self.sub_amount = 0.0
        self.rm_amount = 0.0
        self.sync_amt = 0.0
        self.sync_ratio = 2.0
        self.pd_amt = 0.5
        self.additive_timbre = 0.5
        self.shaper_mode = 'Soft'
        
        self.filter_mode = 'LP'
        self.cutoff = 1.0       
        self.resonance = 0.1   
        self.delay_time = 0.0  
        self.delay_feedback = 0.0
        self.ping_pong = False
        self.reverb_mix = 0.0
        self.bitcrush = 0.0
        self.sidechain = 0.0
        self.chorus_mix = 0.0
        self.formant_mix = 0.0
        self.formant_vowel = 'A'
        
        self.lfo_rate = 0.0
        self.lfo_depth = 0.0
        self.lfo_phase = 0.0
        
        self.smooth_cutoff = 1.0
        self.smooth_volume = 0.8
        self.smooth_reverb = 0.0
        self.smooth_sidechain = 0.0
        self.smooth_chorus = 0.0
        self.smooth_formant = 0.0
        
        self.low_l, self.band_l = 0.0, 0.0
        self.low_r, self.band_r = 0.0, 0.0
        self.f = 1.0
        self.q = 0.1
        
        self.max_delay = int(self.sample_rate * 2.0)
        self.delay_buf_l = np.zeros(self.max_delay, dtype=np.float32)
        self.delay_buf_r = np.zeros(self.max_delay, dtype=np.float32)
        self.delay_ptr = 0
        
        self.reverb = Reverb(sample_rate)
        self.chorus = Chorus(sample_rate)
        self.update_frequencies()
        
        self.playhead_x = 0.0
        self.phase = np.zeros(self.h, dtype=np.float32)
        self.lock = threading.Lock()
        self.last_buffer = np.zeros(1024, dtype=np.float32)
        self.current_amps = np.zeros(self.h, dtype=np.float32)

        try:
            self.stream = sd.OutputStream(samplerate=sample_rate, channels=2, dtype='float32', callback=self.callback)
            self.stream.start()
        except Exception as e:
            print(f"Audio Error: {e}")

    def update_frequencies(self):
        min_f = 130.81 * (2.0 ** self.octave_shift)
        if self.scale == 'pentatonic': intervals = [0, 3, 5, 7, 10]
        elif self.scale == 'major': intervals = [0, 4, 7, 11]
        elif self.scale == 'chromatic': intervals = list(range(12))
        
        if self.scale != 'linear':
            notes = []
            for o in range(3):
                for i in intervals: notes.append(min_f * (2 ** (o + i / 12.0)))
            notes = list(reversed(notes))
            self.freqs = np.array(notes[:self.h], dtype=np.float32)
            self.notes = [freq_to_note_name(f) for f in self.freqs]
        else:
            self.freqs = np.linspace(80, 3000, self.h, dtype=np.float32) * (2.0 ** self.octave_shift)
            self.notes = [f"{int(f)}Hz" for f in self.freqs]
            
        if len(self.freqs) < self.h:
            self.freqs = np.pad(self.freqs, (0, self.h - len(self.freqs)), 'edge')
            
        self.phase_inc = (2.0 * np.pi * self.freqs / self.sample_rate).astype(np.float32)
        
        self.max_ks_len = int(self.sample_rate / 80) 
        self.ks_bufs = np.zeros((self.h, self.max_ks_len), dtype=np.float32)
        self.ks_ptrs = np.zeros(self.h, dtype=np.int32)
        self.ks_lens = np.array([min(self.max_ks_len, max(2, int(self.sample_rate / f))) for f in self.freqs])
        self.prev_amps = np.zeros(self.h, dtype=np.float32)

    def set_grid(self, grid): 
        with self.lock: 
            self.grid = np.copy(grid)
            self.processed_grid = self.process_audio_grid(self.grid)

    # Setters omitted for brevity, identical to previous version
    def set_waveform(self, wf): self.waveform = wf
    def set_scale(self, scale): self.scale = scale; self.update_frequencies()
    def set_volume(self, val): self.volume = val
    def set_octave(self, val): self.octave_shift = val; self.update_frequencies()
    def set_cutoff(self, val): self.cutoff = val
    def set_resonance(self, val): self.resonance = val
    def set_delay_time(self, val): self.delay_time = val
    def set_delay_feedback(self, val): self.delay_feedback = val
    def set_ping_pong(self, state): self.ping_pong = state
    def set_lfo_rate(self, val): self.lfo_rate = val
    def set_lfo_depth(self, val): self.lfo_depth = val
    def set_unison(self, val): self.unison = val
    def set_fm_amount(self, val): self.fm_amount = val
    def set_drive(self, val): self.drive = val
    def set_pluck(self, val): self.pluck_amt = val
    def set_attack(self, val): self.attack = val
    def set_release(self, val): self.release = val
    def set_wt_pos(self, val): self.wt_pos = val
    def set_reverb_mix(self, val): self.reverb_mix = val
    def set_sub_octave(self, val): self.sub_octave = val
    def set_sub_amount(self, val): self.sub_amount = val
    def set_rm_amount(self, val): self.rm_amount = val
    def set_bitcrush(self, val): self.bitcrush = val
    def set_sidechain(self, val): self.sidechain = val
    def set_sync_amt(self, val): self.sync_amt = val
    def set_sync_ratio(self, val): self.sync_ratio = val
    def set_pd_amt(self, val): self.pd_amt = val
    def set_additive_timbre(self, val): self.additive_timbre = val
    def set_shaper_mode(self, mode): self.shaper_mode = mode
    def set_chorus_mix(self, val): self.chorus_mix = val
    def set_filter_mode(self, mode): self.filter_mode = mode
    def set_formant_mix(self, val): self.formant_mix = val
    def set_formant_vowel(self, vowel): self.formant_vowel = vowel
    
    def set_blur(self, val): 
        self.spectral_blur = val
        with self.lock:
            self.processed_grid = self.process_audio_grid(self.grid)
            
    def set_sweep(self, val): 
        self.sweep_duration = val
        with self.lock:
            self.processed_grid = self.process_audio_grid(self.grid)

    def process_audio_grid(self, grid):
        if self.spectral_blur > 0:
            size = int(self.spectral_blur * 6) * 2 + 1
            if size > self.h: size = self.h
            if size <= 1: g = np.copy(grid)
            else:
                kernel = np.hanning(size)
                kernel = kernel / kernel.sum()
                g = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 0, grid).astype(np.float32)
        else:
            g = np.copy(grid)
            
        rel_samples = self.release * self.sample_rate
        if rel_samples > 0.1:
            cols_per_sec = 1.0 / max(0.01, self.sweep_duration)
            rel_cols = max(1.0, rel_samples * cols_per_sec)
            alpha = np.exp(-1.0 / rel_cols)
            for i in range(1, self.w):
                g[:, i] = np.maximum(g[:, i], g[:, i-1] * alpha)
            g[:, 0] = np.maximum(g[:, 0], g[:, -1] * alpha)
            for i in range(1, self.w):
                g[:, i] = np.maximum(g[:, i], g[:, i-1] * alpha)
        return g

    def get_polyblep(self, t, dt):
        pb = np.zeros_like(t)
        mask1 = t < dt
        pb[mask1] = (t[mask1] / dt[mask1]) - 1.0
        mask2 = t > 1.0 - dt
        pb[mask2] = ((t[mask2] - 1.0) / dt[mask2]) + 1.0
        return pb

    def apply_shaper(self, sig):
        d = self.drive
        if self.shaper_mode == 'Soft':
            return np.tanh(sig * d)
        elif self.shaper_mode == 'Hard':
            return np.clip(sig * d, -1, 1)
        elif self.shaper_mode == 'Fold':
            sig = sig * d
            return np.where(sig > 1, 2 - sig, np.where(sig < -1, -2 - sig, sig))
        elif self.shaper_mode == 'Asym':
            return np.tanh(np.where(sig > 0, sig * d, sig * d * 0.5))
        return np.tanh(sig * d)

    def callback(self, outdata, frames, time_info, status):
        with self.lock: 
            grid = self.processed_grid
            
        samples_per_col = (self.sample_rate * self.sweep_duration) / self.w
        playhead_end = self.playhead_x + (frames / samples_per_col)
        x_coords = np.linspace(self.playhead_x, playhead_end, frames)
        x_floor = np.floor(x_coords).astype(int) % self.w
        x_ceil = (x_floor + 1) % self.w
        frac = (x_coords - np.floor(x_coords)).astype(np.float32)
        
        amps = grid[:, x_floor] * (1 - frac) + grid[:, x_ceil] * frac
        amps = np.clip(amps, 0.0, 1.0)
        self.current_amps = amps[:, -1]
        
        col_progress = (x_coords % 1.0)
        attack_cols = max(0.01, self.attack * self.sample_rate * (1.0 / self.sweep_duration))
        att_env = np.minimum(1.0, col_progress / attack_cols)
        decay_rate = 1.0 + (self.pluck_amt * 20.0)
        dec_env = np.exp(-col_progress * decay_rate).astype(np.float32)
        env = att_env * dec_env
        
        left_amps = amps * env[None, :]
        right_amps = amps * env[None, :]
        
        phase_samples = self.phase[:, None] + self.phase_inc[:, None] * np.arange(frames)
        self.phase = np.mod(phase_samples[:, -1], 2.0 * np.pi)
        
        detune = 1.0 + (self.unison * 0.01)
        p2 = phase_samples * detune
        p3 = phase_samples * (2.0 - detune)
        
        t = (phase_samples / (2.0 * np.pi)) % 1.0
        dt = np.maximum(1e-6, self.phase_inc[:, None] / (2.0 * np.pi))
        
        if self.waveform == 'karplus':
            waves = np.zeros((self.h, frames), dtype=np.float32)
            for i in range(frames):
                excite_mask = (amps[:, i] > 0.1) & (self.prev_amps < 0.1)
                excite = np.where(excite_mask, np.random.uniform(-1, 1, self.h), 0.0)
                read_idx = self.ks_ptrs
                next_idx = (self.ks_ptrs + 1) % self.ks_lens
                val = self.ks_bufs[np.arange(self.h), read_idx]
                next_val = self.ks_bufs[np.arange(self.h), next_idx]
                new_val = (val + next_val) * 0.5 * 0.99
                self.ks_bufs[np.arange(self.h), read_idx] = excite + new_val
                self.ks_ptrs = (self.ks_ptrs + 1) % self.ks_lens
                waves[:, i] = val
                self.prev_amps = amps[:, i]
        else:
            if self.waveform == 'sine': waves = np.sin(phase_samples)
            elif self.waveform == 'saw': 
                waves = (2.0 * t - 1.0) - self.get_polyblep(t, dt)
            elif self.waveform == 'square': 
                t2 = (t + 0.5) % 1.0
                saw1 = (2.0 * t - 1.0) - self.get_polyblep(t, dt)
                saw2 = (2.0 * t2 - 1.0) - self.get_polyblep(t2, dt)
                waves = saw1 - saw2
            elif self.waveform == 'triangle': waves = (2/np.pi) * np.arcsin(np.sin(phase_samples))
            elif self.waveform == 'pulse': 
                t2 = (t + 0.25) % 1.0
                saw1 = (2.0 * t - 1.0) - self.get_polyblep(t, dt)
                saw2 = (2.0 * t2 - 1.0) - self.get_polyblep(t2, dt)
                waves = saw1 - saw2
            elif self.waveform == 'super_saw': 
                t2_p = (p2 / (2.0 * np.pi)) % 1.0
                t3_p = (p3 / (2.0 * np.pi)) % 1.0
                dt2 = np.maximum(1e-6, (self.phase_inc[:, None] * detune) / (2.0 * np.pi))
                dt3 = np.maximum(1e-6, (self.phase_inc[:, None] * (2.0 - detune)) / (2.0 * np.pi))
                s1 = (2.0 * t - 1.0) - self.get_polyblep(t, dt)
                s2 = (2.0 * t2_p - 1.0) - self.get_polyblep(t2_p, dt2)
                s3 = (2.0 * t3_p - 1.0) - self.get_polyblep(t3_p, dt3)
                waves = (s1 + s2 + s3) / 3.0
            elif self.waveform == 'fm_bell': waves = np.sin(phase_samples + self.fm_amount * np.sin(phase_samples * 3.5))
            elif self.waveform == 'theremin': 
                vibrato = np.sin(np.arange(frames) * 2 * np.pi * 5.0 / self.sample_rate) * 0.5
                waves = np.sin(phase_samples + vibrato[None, :])
            elif self.waveform == 'organ': waves = (np.sin(phase_samples) + 0.5*np.sin(2*phase_samples) + 0.25*np.sin(3*phase_samples) + 0.125*np.sin(4*phase_samples)) / 1.875
            elif self.waveform == 'noise': waves = np.random.uniform(-1, 1, phase_samples.shape).astype(np.float32)
            elif self.waveform == 'phase_dist': 
                pd_curve = t + (self.pd_amt * np.sin(2 * np.pi * t))
                waves = np.sin(2 * np.pi * pd_curve % 1.0)
            elif self.waveform == 'additive': 
                waves = np.zeros_like(phase_samples)
                peak_harmonic = 1 + self.additive_timbre * 15
                for h in range(1, 17):
                    weight = np.exp(-0.5 * ((h - peak_harmonic) / (1 + self.additive_timbre*2))**2)
                    waves += weight * np.sin(phase_samples * h)
                max_amp = np.max(np.abs(waves))
                if max_amp > 0: waves = waves / max_amp
            elif self.waveform == 'wavetable': 
                wt_pos = np.clip(self.wt_pos, 0, 1) * 3.0
                idx = int(wt_pos)
                frac_wt = wt_pos - idx
                w1 = np.sin(phase_samples)
                w2 = (2/np.pi) * np.arcsin(np.sin(phase_samples))
                w3 = 2.0 * (t - np.floor(t + 0.5))
                w4 = np.where(t < 0.5, 1.0, -1.0)
                wt_list = [w1, w2, w3, w4]
                waves = wt_list[idx] * (1 - frac_wt) + wt_list[min(idx+1, 3)] * frac_wt
                
        if self.sync_amt > 0 and self.waveform != 'karplus':
            ratio = 1.0 + (self.sync_ratio * 3.0)
            t_sync = (t * ratio) % 1.0
            dt_sync = dt * ratio
            synced_wave = (2.0 * t_sync - 1.0) - self.get_polyblep(t_sync, dt_sync)
            waves = (waves * (1.0 - self.sync_amt)) + (synced_wave * self.sync_amt)
            
        if self.sub_amount > 0 and self.waveform != 'karplus':
            sub_mult = 2.0 ** abs(self.sub_octave) if self.sub_octave < 0 else 1.0 / (2.0 ** self.sub_octave)
            sub_t = (t * sub_mult) % 1.0
            sub_dt = np.maximum(1e-6, dt * sub_mult)
            sub_t2 = (sub_t + 0.5) % 1.0
            sub_sq = ((2.0 * sub_t - 1.0) - self.get_polyblep(sub_t, sub_dt)) - ((2.0 * sub_t2 - 1.0) - self.get_polyblep(sub_t2, sub_dt))
            waves = (waves * (1.0 - self.sub_amount)) + (sub_sq * self.sub_amount)
            
        if self.rm_amount > 0 and self.waveform != 'karplus':
            rm_freq = 220.0 * (2.0 ** self.octave_shift)
            rm_mod = np.sin(2.0 * np.pi * np.arange(frames) * rm_freq / self.sample_rate)
            waves = (waves * (1.0 - self.rm_amount)) + (waves * rm_mod[None, :] * self.rm_amount)
            
        if self.waveform != 'karplus':
            waves = self.apply_shaper(waves)
            
        left_audio = np.sum(waves * left_amps, axis=0)
        right_audio = np.sum(waves * right_amps, axis=0)
        
        active_count = np.sum(amps > 0.1, axis=0)
        left_audio = np.where(active_count > 0, left_audio / np.sqrt(active_count + 1), 0)
        right_audio = np.where(active_count > 0, right_audio / np.sqrt(active_count + 1), 0)
        
        self.smooth_volume = self.smooth_volume * 0.95 + self.volume * 0.05
        self.smooth_reverb = self.smooth_reverb * 0.95 + self.reverb_mix * 0.05
        self.smooth_sidechain = self.smooth_sidechain * 0.95 + self.sidechain * 0.05
        self.smooth_chorus = self.smooth_chorus * 0.95 + self.chorus_mix * 0.05
        self.smooth_formant = self.smooth_formant * 0.95 + self.formant_mix * 0.05
        
        if self.smooth_sidechain > 0.01:
            duck_phase = (x_coords % 1.0)
            env = np.exp(-duck_phase * (5.0 + (1.0 - self.smooth_sidechain) * 20.0))
            duck_amp = 1.0 - (env * self.smooth_sidechain)
            left_audio *= duck_amp
            right_audio *= duck_amp
        
        out_left = np.zeros(frames, dtype=np.float32)
        out_right = np.zeros(frames, dtype=np.float32)
        
        vowels = {
            'A': [(800, 0.1), (1150, 0.2), (2900, 0.1)],
            'E': [(400, 0.1), (2000, 0.2), (2800, 0.1)],
            'I': [(350, 0.1), (2300, 0.2), (3000, 0.1)],
            'O': [(450, 0.1), (900, 0.2), (2800, 0.1)],
            'U': [(325, 0.1), (700, 0.2), (2500, 0.1)]
        }
        formants = vowels.get(self.formant_vowel, vowels['A'])
        
        lfo_inc = 2.0 * np.pi * self.lfo_rate / self.sample_rate
        
        for i in range(frames):
            lfo_val = np.sin(self.lfo_phase) * self.lfo_depth
            self.lfo_phase += lfo_inc
            if self.lfo_phase > 2.0 * np.pi: self.lfo_phase -= 2.0 * np.pi
            
            mod_cutoff = np.clip(self.cutoff + lfo_val, 0.0, 1.0)
            self.smooth_cutoff = self.smooth_cutoff * 0.95 + mod_cutoff * 0.05
            
            freq = 30.0 * (18000.0 / 30.0) ** self.smooth_cutoff
            f_target = 2.0 * np.sin(np.pi * freq / self.sample_rate)
            q_target = 1.0 - np.clip(self.resonance * 0.95, 0.0, 0.95)
            self.f = self.f * 0.99 + f_target * 0.01
            self.q = self.q * 0.99 + q_target * 0.01
            
            high_l = left_audio[i] - self.low_l - self.q * self.band_l
            band_l = np.tanh(self.f * high_l + self.band_l) 
            low_l = self.f * band_l + self.low_l
            self.low_l = low_l
            self.band_l = band_l
            
            high_r = right_audio[i] - self.low_r - self.q * self.band_r
            band_r = np.tanh(self.f * high_r + self.band_r)
            low_r = self.f * band_r + self.low_r
            self.low_r = low_r
            self.band_r = band_r
            
            if self.filter_mode == 'LP':
                out_left[i] = low_l; out_right[i] = low_r
            elif self.filter_mode == 'HP':
                out_left[i] = high_l; out_right[i] = high_r
            elif self.filter_mode == 'BP':
                out_left[i] = band_l; out_right[i] = band_r
            elif self.filter_mode == 'Notch':
                out_left[i] = low_l + high_l; out_right[i] = low_r + high_r

        if self.bitcrush > 0:
            bits = 2 + (1.0 - self.bitcrush) * 14
            step = 2 ** (bits - 1)
            out_left = np.floor(out_left * step) / step
            out_right = np.floor(out_right * step) / step
            sr_div = 1 + int(self.bitcrush * 15)
            idx = np.arange(0, frames, sr_div)
            interp_idx = np.arange(frames)
            out_left = np.interp(interp_idx, idx, out_left[idx])
            out_right = np.interp(interp_idx, idx, out_right[idx])

        if self.smooth_chorus > 0.01:
            out_left, out_right = self.chorus.process_block(out_left, out_right, self.smooth_chorus)

        delay_samples = self.delay_time * self.sample_rate
        if delay_samples > 1 and delay_samples < self.max_delay - 1:
            read_ptr = (self.delay_ptr - delay_samples + self.max_delay) % self.max_delay
            base = np.floor(read_ptr).astype(int)
            frac_d = read_ptr - base
            next_base = (base + 1) % self.max_delay
            
            read_indices = (base + np.arange(frames)) % self.max_delay
            next_indices = (next_base + np.arange(frames)) % self.max_delay
            
            delayed_l = self.delay_buf_l[read_indices] * (1 - frac_d) + self.delay_buf_l[next_indices] * frac_d
            delayed_r = self.delay_buf_r[read_indices] * (1 - frac_d) + self.delay_buf_r[next_indices] * frac_d
            
            if self.ping_pong:
                fb_l = out_left + delayed_r * self.delay_feedback
                fb_r = out_right + delayed_l * self.delay_feedback
            else:
                fb_l = out_left + delayed_l * self.delay_feedback
                fb_r = out_right + delayed_r * self.delay_feedback
            
            write_indices = (self.delay_ptr + np.arange(frames)) % self.max_delay
            self.delay_buf_l[write_indices] = fb_l
            self.delay_buf_r[write_indices] = fb_r
                
            self.delay_ptr = (self.delay_ptr + frames) % self.max_delay
            
            out_left = out_left + delayed_l * 0.6
            out_right = out_right + delayed_r * 0.6
            
        if self.smooth_reverb > 0.01:
            rev_l, rev_r = self.reverb.process_block(out_left, out_right)
            out_left = (out_left * (1.0 - self.smooth_reverb)) + (rev_l * self.smooth_reverb)
            out_right = (out_right * (1.0 - self.smooth_reverb)) + (rev_r * self.smooth_reverb)
        
        outdata[:, 0] = np.tanh(out_left * self.smooth_volume * 1.5)
        outdata[:, 1] = np.tanh(out_right * self.smooth_volume * 1.5)
        
        self.last_buffer = (out_left + out_right) * 0.5
        self.playhead_x = playhead_end % self.w

# --- 4. MIDI Export Logic ---
def export_midi(filename, grid, freqs, sweep_duration):
    ticks_per_col = int((sweep_duration / grid.shape[1]) * 480 * 2)
    def var_len(value):
        buffer = [value & 0x7F]
        while value >> 7:
            value >>= 7
            buffer.insert(0, (value & 0x7F) | 0x80)
        return bytes(buffer)

    track = []
    for col in range(grid.shape[1]):
        active_notes = []
        for row in range(grid.shape[0]):
            if grid[row, col] > 0.5: # Active note
                midi_note = freq_to_midi(freqs[row])
                active_notes.append(midi_note)
                track.append((0, bytes([0x90, midi_note, 100])))
        track.append((ticks_per_col, b''))
        for note in active_notes:
            track.append((0, bytes([0x80, note, 0])))
    track.append((0, bytes([0xFF, 0x2F, 0x00])))
    
    track_bytes = b''
    for delta, event in track:
        track_bytes += var_len(delta) + event
        
    header = b'MThd' + struct.pack('>IHHH', 6, 0, 1, 480)
    track_header = b'MTrk' + struct.pack('>I', len(track_bytes))
    
    with open(filename, 'wb') as f:
        f.write(header + track_header + track_bytes)

# --- 5. Visual Widgets ---
class SpectrogramWidget(QWidget):
    def __init__(self, grid_w, grid_h):
        super().__init__()
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.setMinimumSize(500, 300)
        self.grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.lut = self.get_lut("Inferno")
        self.playhead_x = 0.0
        self.last_pos = None
        self.notes = []
        self.view_mode = "2D Plane"
        self.show_grid = False
        self.brush_size = 2
        self.cam_yaw = -0.5
        self.cam_pitch = 0.6
        self.armed_pattern = None
        self.setMouseTracking(True)

    def get_lut(self, name):
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if name == "Inferno":
                r = min(255, int(t * 255 * 2.0)); g = max(0, int((t - 0.4) * 255 * 1.6)); b = max(0, int(np.sin(t * np.pi) * 180))
                lut[i] = [r, g, b]
            elif name == "Viridis":
                r = int(max(0, (t - 0.5) * 2.0 * 255)); g = int(np.sin(t * np.pi) * 220 + 30); b = int(max(0, (1.0 - t * 1.5) * 255))
                lut[i] = [r, g, b]
            elif name == "Plasma":
                lut[i] = [255, int(t * 200), int(t * 255)]
            elif name == "Spectrum":
                if t < 0.25: lut[i] = [0, int(t*4*255), 255]
                elif t < 0.5: lut[i] = [0, 255, int(255 - (t-0.25)*4*255)]
                elif t < 0.75: lut[i] = [int((t-0.5)*4*255), 255, 0]
                else: lut[i] = [255, int(255 - (t-0.75)*4*255), 0]
        return lut

    def update_grid(self, grid): self.grid = grid; self.update()
    def update_playhead(self, x): self.playhead_x = x; self.update()
    def set_colormap(self, name): self.lut = self.get_lut(name); self.update()
    def set_view_mode(self, mode): self.view_mode = mode; self.update()
    def set_grid_overlay(self, state): self.show_grid = state; self.update()
    def set_brush_size(self, size): self.brush_size = size
    def set_cam_yaw(self, val): self.cam_yaw = val; self.update()
    def set_cam_pitch(self, val): self.cam_pitch = val; self.update()
    
    def arm_pattern(self, pattern_name):
        if pattern_name == "None":
            self.armed_pattern = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.armed_pattern = pattern_name
            self.setCursor(Qt.CursorShape.CrossCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(10, 10, 10))
        if self.view_mode == "3D Waterfall": self.draw_3d(painter)
        else: self.draw_2d(painter)

    def draw_2d(self, painter):
        pixels = np.ascontiguousarray(self.lut[(self.grid * 255).astype(np.uint8)])
        image = QImage(pixels.data, self.grid_w, self.grid_h, self.grid_w * 3, QImage.Format.Format_RGB888)
        rect = QRectF(40, 0, self.width()-40, self.height())
        painter.drawImage(rect, image)
        if self.show_grid:
            painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
            for i in range(0, self.grid_w, 8):
                x = 40 + (i / self.grid_w) * rect.width()
                painter.drawLine(x, 0, x, self.height())
            for i in range(0, self.grid_h, 4):
                y = (i / self.grid_h) * rect.height()
                painter.drawLine(40, y, self.width(), y)
        painter.setPen(QColor(120, 120, 120))
        font = painter.font(); font.setPointSize(8); painter.setFont(font)
        for i, note in enumerate(self.notes):
            if i % 4 == 0 or i == len(self.notes)-1:
                y = int((i + 0.5) * (rect.height() / self.grid_h))
                painter.drawText(2, y + 4, note)
        playhead_px = int((self.playhead_x / self.grid_w) * rect.width()) + 40
        gradient = QLinearGradient(playhead_px - 15, 0, playhead_px + 15, 0)
        gradient.setColorAt(0, QColor(255, 255, 255, 0))
        gradient.setColorAt(0.5, QColor(255, 255, 255, 120))
        gradient.setColorAt(1, QColor(255, 255, 255, 0))
        painter.fillRect(QRectF(playhead_px - 15, 0, 30, self.height()), QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        painter.drawLine(playhead_px, 0, playhead_px, self.height())

    def draw_3d(self, painter):
        w, h = self.width(), self.height()
        cx, cy = w / 2, h * 0.65
        scale_x = w / (self.grid_w + self.grid_h)
        scale_y = scale_x * 0.6
        height_scale = h * 0.2
        sin_yaw = np.sin(self.cam_yaw); cos_yaw = np.cos(self.cam_yaw)
        sin_pitch = np.sin(self.cam_pitch); cos_pitch = np.cos(self.cam_pitch)
        y_coords, x_coords = np.mgrid[0:self.grid_h, 0:self.grid_w]
        flat_y = y_coords.ravel()
        flat_x = x_coords.ravel()
        vals = self.grid.ravel()
        xt = flat_x - self.grid_w / 2
        yt = flat_y - self.grid_h / 2
        x_rot = xt * cos_yaw - yt * sin_yaw
        y_rot = xt * sin_yaw + yt * cos_yaw
        depth = y_rot * sin_pitch
        sort_idx = np.argsort(depth)
        def project(x, y, z):
            y_pitch = y * cos_pitch - z * sin_pitch
            z_pitch = y * sin_pitch + z * cos_pitch
            sx = cx + x * scale_x
            sy = cy - y_pitch * scale_y - z_pitch * height_scale
            return sx, sy
        for i in sort_idx:
            v = vals[i]
            if v < 0.05: continue
            x = flat_x[i]; y = flat_y[i]
            xr = x_rot[i]; yr = y_rot[i]
            color_idx = int(v * 255)
            c = self.lut[color_idx]
            color = QColor(c[0], c[1], c[2], 220)
            p1 = project(xr, yr, 0); p2 = project(xr+1, yr, 0)
            p3 = project(xr+1, yr+1, 0); p4 = project(xr, yr+1, 0)
            t1 = project(xr, yr, v); t2 = project(xr+1, yr, v)
            t3 = project(xr+1, yr+1, v); t4 = project(xr, yr+1, v)
            if x == 0 or self.grid[y, x-1] < 0.05:
                painter.setBrush(color.darker(120)); painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon(QPolygonF([QPointF(*p) for p in [p1, p2, t2, t1]]))
            if y == self.grid_h - 1 or self.grid[y+1, x] < 0.05:
                painter.setBrush(color.darker(150)); painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon(QPolygonF([QPointF(*p) for p in [p4, p3, t3, t4]]))
            painter.setBrush(color); painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(QPolygonF([QPointF(*p) for p in [t1, t2, t3, t4]]))
        ph_x = self.playhead_x - self.grid_w / 2
        y1 = -self.grid_h / 2
        y2 = self.grid_h / 2
        xr1 = ph_x * cos_yaw - y1 * sin_yaw
        yr1 = ph_x * sin_yaw + y1 * cos_yaw
        xr2 = ph_x * cos_yaw - y2 * sin_yaw
        yr2 = ph_x * sin_yaw + y2 * cos_yaw
        pt1 = project(xr1, yr1, 0)
        pt2 = project(xr2, yr2, 2.0)
        painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
        painter.drawLine(QPointF(*pt1), QPointF(*pt2))

    def mousePressEvent(self, event):
        if self.view_mode != "2D Plane": return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.armed_pattern:
                pos = self.widget_to_grid(event.position())
                self.parentWidget().parentWidget().place_pattern_at(self.armed_pattern, pos)
                self.armed_pattern = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.parentWidget().parentWidget().pattern_combo.setCurrentText("None")
            else:
                self.last_pos = self.widget_to_grid(event.position())
                self.draw_at(self.last_pos)
        elif event.button() == Qt.MouseButton.RightButton:
            self.grid[:] = 0

    def mouseMoveEvent(self, event):
        if self.view_mode != "2D Plane" or self.armed_pattern: return
        if event.buttons() & Qt.MouseButton.LeftButton:
            pos = self.widget_to_grid(event.position())
            if self.last_pos: self.draw_line(self.last_pos, pos)
            self.last_pos = pos

    def widget_to_grid(self, pos):
        rect = QRectF(40, 0, self.width()-40, self.height())
        x = int((pos.x() - 40) / rect.width() * self.grid_w)
        y = int(pos.y() / rect.height() * self.grid_h)
        return max(0, min(self.grid_w - 1, x)), max(0, min(self.grid_h - 1, y))

    def draw_at(self, pos):
        x, y = pos
        s = self.brush_size
        for dx in range(-s, s+1):
            for dy in range(-s, s+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    self.grid[ny, nx] = 1.0

    def draw_line(self, p1, p2):
        if not p1: return
        x1, y1 = p1; x2, y2 = p2
        steps = max(abs(x2 - x1), abs(y2 - y1))
        for i in range(steps + 1):
            t = i / max(1, steps)
            self.draw_at((int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))))

class SpectrumBarsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.amps = np.zeros(GRID_H, dtype=np.float32)
        self.notes = []
        
    def update_amps(self, amps, notes):
        self.amps = amps
        self.notes = notes
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor(15, 15, 15))
        bar_w = rect.width() / GRID_H
        for i in range(GRID_H):
            val = self.amps[i]
            h = val * (rect.height() - 10)
            x = i * bar_w
            y = rect.height() - h
            gradient = QLinearGradient(x, y, x, rect.height())
            gradient.setColorAt(0, QColor(255, 255, 255, 220))
            gradient.setColorAt(0.5, QColor(255, 100, 0, 200))
            gradient.setColorAt(1, QColor(100, 0, 0, 150))
            painter.fillRect(QRectF(x+1, y, bar_w-2, h), QBrush(gradient))

class OscilloscopeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(80)
        self.buffer = np.zeros(1024, dtype=np.float32)
        
    def update_buffer(self, buf): self.buffer = buf; self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor(10, 10, 10))
        painter.setPen(QColor(30, 30, 30))
        mid_y = rect.height() / 2
        painter.drawLine(0, int(mid_y), rect.width(), int(mid_y))
        pen = QPen(QColor(0, 255, 150, 200)); pen.setWidth(2); painter.setPen(pen)
        x_coords = np.linspace(0, len(self.buffer)-1, rect.width())
        y_vals = np.interp(x_coords, np.arange(len(self.buffer)), self.buffer)
        points = [QPointF(x, mid_y - (y * mid_y * 0.9)) for x, y in zip(x_coords, y_vals)]
        if points: painter.drawPolyline(points)

# --- 6. XY Pad Widget ---
class XYPad(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(180)
        self.x_val = 1.0
        self.y_val = 0.1
        self.setMouseTracking(True)
        
    def set_values(self, x, y):
        self.x_val = max(0.0, min(1.0, x))
        self.y_val = max(0.0, min(1.0, y))
        self.update()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.update_from_pos(event.position())
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.update_from_pos(event.position())
            
    def update_from_pos(self, pos):
        rx = max(0.0, min(1.0, pos.x() / self.width()))
        ry = max(0.0, min(1.0, 1.0 - (pos.y() / self.height())))
        self.set_values(rx, ry)
        self.parentWidget().parentWidget().update_xy_filter(rx, ry)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor(50, 10, 10))
        gradient.setColorAt(1, QColor(15, 15, 15))
        painter.fillRect(rect, QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        for i in range(1, 4):
            painter.drawLine(int(i * rect.width() / 4), 0, int(i * rect.width() / 4), rect.height())
            painter.drawLine(0, int(i * rect.height() / 4), rect.width(), int(i * rect.height() / 4))
        hx = int(self.x_val * rect.width())
        hy = int((1.0 - self.y_val) * rect.height())
        painter.setPen(QPen(QColor(0, 255, 255), 2))
        painter.setBrush(QColor(0, 255, 255, 150))
        painter.drawEllipse(QPointF(hx, hy), 8, 8)
        painter.setPen(QColor(120, 120, 120))
        painter.drawText(5, 15, "Resonance (Y)")
        painter.drawText(rect.width() - 60, rect.height() - 5, "Cutoff (X)")

# --- 7. Helper Widgets ---
class LabeledSlider(QWidget):
    def __init__(self, label, min_val, max_val, init_val, suffix=""):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = QHBoxLayout()
        self.label = QLabel(label)
        self.value_label = QLabel(str(init_val) + suffix)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.label)
        header.addWidget(self.value_label)
        layout.addLayout(header)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(init_val)
        layout.addWidget(self.slider)
        self.suffix = suffix
        self.slider.valueChanged.connect(self.update_label)
        
    def update_label(self, val):
        self.value_label.setText(str(val) + self.suffix)

# --- 8. Main Application ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CA Spectrogram Studio - Advanced Generative Engine")
        self.resize(1200, 750)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.ca = CAEngine(GRID_W, GRID_H)
        self.audio = AudioEngine(SAMPLE_RATE, (GRID_H, GRID_W))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)
        
        visual_widget = QWidget()
        visual_layout = QVBoxLayout(visual_widget)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(5)
        
        self.spectrogram = SpectrogramWidget(GRID_W, GRID_H)
        self.spectrogram.set_brush_size(2)
        visual_layout.addWidget(self.spectrogram, stretch=4)
        self.bars_widget = SpectrumBarsWidget()
        visual_layout.addWidget(self.bars_widget, stretch=1)
        self.oscilloscope = OscilloscopeWidget()
        visual_layout.addWidget(self.oscilloscope, stretch=1)
        main_layout.addWidget(visual_widget, stretch=1)
        
        sidebar = QWidget()
        sidebar.setMaximumWidth(340)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        
        # TAB 1: Seq & Synth
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        
        transport_group = QGroupBox("Transport")
        transport_layout = QVBoxLayout(transport_group)
        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.setChecked(True)
        self.play_btn.clicked.connect(self.toggle_play)
        btn_row.addWidget(self.play_btn)
        self.random_btn = QPushButton("Random")
        self.random_btn.clicked.connect(self.randomize)
        btn_row.addWidget(self.random_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        btn_row.addWidget(self.clear_btn)
        transport_layout.addLayout(btn_row)
        
        self.sweep_slider = LabeledSlider("Sweep Time", 1, 100, 40, "")
        self.sweep_slider.slider.valueChanged.connect(self.on_sweep_changed)
        self.on_sweep_changed(40)
        transport_layout.addWidget(self.sweep_slider)
        
        self.speed_slider = LabeledSlider("CA Speed", 1, 60, 15, " fps")
        self.speed_slider.slider.valueChanged.connect(self.update_speed)
        transport_layout.addWidget(self.speed_slider)
        tab1_layout.addWidget(transport_group)

        ca_group = QGroupBox("Cellular Automata Engine")
        ca_layout = QVBoxLayout(ca_group)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Rule:"))
        self.rule_combo = QComboBox()
        self.rule_combo.addItems(["Custom", "HighLife", "Conway", "Seeds", "Maze", "Replicator", "Day & Night", "Gnarl", "Assimilation", "Coral", "Star Wars (Gen)", "Bloom (Gen)", "Brians Brain (Gen)", "Burst (Gen)", "Wireworld", "Langton's Ant", "Procedural Noise"])
        self.rule_combo.currentTextChanged.connect(self.on_rule_changed)
        row.addWidget(self.rule_combo)
        ca_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Custom B/S/C:"))
        self.custom_rule_edit = QLineEdit("B3/S23/2")
        self.custom_rule_edit.setToolTip("Standard Life: B3/S23/2. Generations: B2/S/3. Wireworld is specific.")
        self.custom_rule_edit.returnPressed.connect(self.on_custom_rule)
        row.addWidget(self.custom_rule_edit)
        ca_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["None", "Glider", "LWSS", "Acorn", "R-Pentomino", "Pulsar", "Pentadecathlon", "Gosper Glider Gun", "Blinker", "Toad", "WW Clock (Fast)", "WW XOR Gate"])
        self.pattern_combo.currentTextChanged.connect(self.on_pattern_changed)
        row.addWidget(self.pattern_combo)
        ca_layout.addLayout(row)
        
        row = QHBoxLayout()
        self.drop_btn = QPushButton("Arm Placement")
        self.drop_btn.clicked.connect(self.arm_pattern)
        row.addWidget(self.drop_btn)
        self.sprinkle_btn = QPushButton("Sprinkle!")
        self.sprinkle_btn.setToolTip("Randomly tosses multiple instances of the selected pattern across the grid.")
        self.sprinkle_btn.clicked.connect(self.sprinkle_patterns)
        row.addWidget(self.sprinkle_btn)
        ca_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Rotation:"))
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(["0°", "90°", "180°", "270°"])
        row.addWidget(self.rotation_combo)
        ca_layout.addLayout(row)
        
        self.chaos_slider = LabeledSlider("Chaos (Mutation)", 0, 100, 0, "%")
        self.chaos_slider.slider.valueChanged.connect(lambda v: setattr(self.ca, 'mutation_rate', v / 1000.0))
        ca_layout.addWidget(self.chaos_slider)
        
        self.midi_btn = QPushButton("Export Grid to MIDI")
        self.midi_btn.clicked.connect(self.export_to_midi)
        ca_layout.addWidget(self.midi_btn)
        tab1_layout.addWidget(ca_group)

        synth_group = QGroupBox("Synthesis Engine")
        synth_layout = QVBoxLayout(synth_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Wave:"))
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["sine", "saw", "square", "triangle", "pulse", "super_saw", "fm_bell", "theremin", "organ", "noise", "wavetable", "phase_dist", "additive", "karplus"])
        self.wave_combo.currentTextChanged.connect(self.audio.set_waveform)
        row.addWidget(self.wave_combo)
        row.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["pentatonic", "major", "chromatic", "linear"])
        self.scale_combo.currentTextChanged.connect(self.audio.set_scale)
        row.addWidget(self.scale_combo)
        synth_layout.addLayout(row)
        
        self.vol_slider = LabeledSlider("Volume", 0, 100, 80, "%")
        self.vol_slider.slider.valueChanged.connect(lambda v: self.audio.set_volume(v / 100.0))
        synth_layout.addWidget(self.vol_slider)
        
        self.octave_slider = LabeledSlider("Octave", -2, 2, 0, "")
        self.octave_slider.slider.valueChanged.connect(lambda v: self.audio.set_octave(v))
        synth_layout.addWidget(self.octave_slider)
        
        self.attack_slider = LabeledSlider("Attack", 0, 100, 5, "%")
        self.attack_slider.slider.valueChanged.connect(lambda v: self.audio.set_attack((v / 100.0) * 0.5))
        synth_layout.addWidget(self.attack_slider)
        
        self.pluck_slider = LabeledSlider("Pluck/Decay", 0, 100, 50, "%")
        self.pluck_slider.slider.valueChanged.connect(lambda v: self.audio.set_pluck(v / 100.0))
        synth_layout.addWidget(self.pluck_slider)
        
        self.release_slider = LabeledSlider("Release", 0, 100, 10, "%")
        self.release_slider.slider.valueChanged.connect(lambda v: self.audio.set_release((v / 100.0) * 1.0))
        synth_layout.addWidget(self.release_slider)
        
        self.drive_slider = LabeledSlider("Drive", 1, 100, 10, "%")
        self.drive_slider.slider.valueChanged.connect(lambda v: self.audio.set_drive((v / 100.0) * 5.0))
        synth_layout.addWidget(self.drive_slider)
        
        self.unison_slider = LabeledSlider("Unison (Detune)", 0, 100, 10, "%")
        self.unison_slider.slider.valueChanged.connect(lambda v: self.audio.set_unison(v / 100.0))
        synth_layout.addWidget(self.unison_slider)
        
        self.fm_slider = LabeledSlider("FM Amount", 0, 100, 20, "%")
        self.fm_slider.slider.valueChanged.connect(lambda v: self.audio.set_fm_amount((v / 100.0) * 5.0))
        synth_layout.addWidget(self.fm_slider)
        
        self.wt_slider = LabeledSlider("Wavetable Morph", 0, 100, 0, "%")
        self.wt_slider.slider.valueChanged.connect(lambda v: self.audio.set_wt_pos(v / 100.0))
        synth_layout.addWidget(self.wt_slider)
        
        self.blur_slider = LabeledSlider("Harmonics", 0, 100, 0, "%")
        self.blur_slider.slider.valueChanged.connect(lambda v: self.audio.set_blur(v / 100.0))
        synth_layout.addWidget(self.blur_slider)
        tab1_layout.addWidget(synth_group)
        
        layer_group = QGroupBox("Layering & Modulation")
        layer_layout = QVBoxLayout(layer_group)
        self.sub_amt_slider = LabeledSlider("Sub Osc Amount", 0, 100, 0, "%")
        self.sub_amt_slider.slider.valueChanged.connect(lambda v: self.audio.set_sub_amount(v / 100.0))
        layer_layout.addWidget(self.sub_amt_slider)
        self.sub_oct_slider = LabeledSlider("Sub Octave", -2, 0, -1, "")
        self.sub_oct_slider.slider.valueChanged.connect(lambda v: self.audio.set_sub_octave(v))
        layer_layout.addWidget(self.sub_oct_slider)
        self.rm_slider = LabeledSlider("Ring Mod Amount", 0, 100, 0, "%")
        self.rm_slider.slider.valueChanged.connect(lambda v: self.audio.set_rm_amount(v / 100.0))
        layer_layout.addWidget(self.rm_slider)
        self.sync_amt_slider = LabeledSlider("Hard Sync Amount", 0, 100, 0, "%")
        self.sync_amt_slider.slider.valueChanged.connect(lambda v: self.audio.set_sync_amt(v / 100.0))
        layer_layout.addWidget(self.sync_amt_slider)
        self.sync_ratio_slider = LabeledSlider("Sync Ratio", 0, 100, 33, "%")
        self.sync_ratio_slider.slider.valueChanged.connect(lambda v: self.audio.set_sync_ratio(v / 100.0))
        layer_layout.addWidget(self.sync_ratio_slider)
        self.pd_slider = LabeledSlider("Phase Distortion", 0, 100, 50, "%")
        self.pd_slider.slider.valueChanged.connect(lambda v: self.audio.set_pd_amt(v / 100.0))
        layer_layout.addWidget(self.pd_slider)
        self.add_slider = LabeledSlider("Additive Timbre", 0, 100, 50, "%")
        self.add_slider.slider.valueChanged.connect(lambda v: self.audio.set_additive_timbre(v / 100.0))
        layer_layout.addWidget(self.add_slider)
        tab1_layout.addWidget(layer_group)
        tab1_layout.addStretch()
        
        # TAB 2: FX
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        filter_group = QGroupBox("Filter (Multi-Mode SVF)")
        filter_layout = QVBoxLayout(filter_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self.filter_mode_combo = QComboBox()
        self.filter_mode_combo.addItems(["LP", "HP", "BP", "Notch"])
        self.filter_mode_combo.currentTextChanged.connect(self.audio.set_filter_mode)
        row.addWidget(self.filter_mode_combo)
        filter_layout.addLayout(row)
        self.xy_pad = XYPad()
        filter_layout.addWidget(self.xy_pad)
        tab2_layout.addWidget(filter_group)
        
        formant_group = QGroupBox("Formant Filter (Vocals)")
        formant_layout = QVBoxLayout(formant_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Vowel:"))
        self.vowel_combo = QComboBox()
        self.vowel_combo.addItems(["A", "E", "I", "O", "U"])
        self.vowel_combo.currentTextChanged.connect(self.audio.set_formant_vowel)
        row.addWidget(self.vowel_combo)
        formant_layout.addLayout(row)
        self.formant_slider = LabeledSlider("Formant Mix", 0, 100, 0, "%")
        self.formant_slider.slider.valueChanged.connect(lambda v: self.audio.set_formant_mix(v / 100.0))
        formant_layout.addWidget(self.formant_slider)
        tab2_layout.addWidget(formant_group)
        
        lfo_group = QGroupBox("LFO (Filter Modulator)")
        lfo_layout = QVBoxLayout(lfo_group)
        self.lfo_rate_slider = LabeledSlider("LFO Rate", 0, 100, 0, "%")
        self.lfo_rate_slider.slider.valueChanged.connect(lambda v: self.audio.set_lfo_rate((v / 100.0) * 10.0))
        lfo_layout.addWidget(self.lfo_rate_slider)
        self.lfo_depth_slider = LabeledSlider("LFO Depth", 0, 100, 0, "%")
        self.lfo_depth_slider.slider.valueChanged.connect(lambda v: self.audio.set_lfo_depth(v / 100.0))
        lfo_layout.addWidget(self.lfo_depth_slider)
        tab2_layout.addWidget(lfo_group)
        
        shaper_group = QGroupBox("Waveshaper")
        shaper_layout = QVBoxLayout(shaper_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self.shaper_combo = QComboBox()
        self.shaper_combo.addItems(["Soft", "Hard", "Fold", "Asym"])
        self.shaper_combo.currentTextChanged.connect(self.audio.set_shaper_mode)
        row.addWidget(self.shaper_combo)
        shaper_layout.addLayout(row)
        tab2_layout.addWidget(shaper_group)
        
        delay_group = QGroupBox("Stereo Delay")
        delay_layout = QVBoxLayout(delay_group)
        self.delay_time_slider = LabeledSlider("Delay Time", 0, 100, 0, "%")
        self.delay_time_slider.slider.valueChanged.connect(lambda v: self.audio.set_delay_time((v / 100.0) * 2.0))
        delay_layout.addWidget(self.delay_time_slider)
        self.delay_fb_slider = LabeledSlider("Delay Feedback", 0, 100, 0, "%")
        self.delay_fb_slider.slider.valueChanged.connect(lambda v: self.audio.set_delay_feedback(v / 100.0))
        delay_layout.addWidget(self.delay_fb_slider)
        self.ping_check = QCheckBox("Ping-Pong Echo")
        self.ping_check.toggled.connect(self.audio.set_ping_pong)
        delay_layout.addWidget(self.ping_check)
        tab2_layout.addWidget(delay_group)
        
        mod_group = QGroupBox("Modulation & Reverb")
        mod_layout = QVBoxLayout(mod_group)
        self.chorus_slider = LabeledSlider("Chorus / Dimension", 0, 100, 0, "%")
        self.chorus_slider.slider.valueChanged.connect(lambda v: self.audio.set_chorus_mix(v / 100.0))
        mod_layout.addWidget(self.chorus_slider)
        self.reverb_slider = LabeledSlider("Reverb Mix", 0, 100, 0, "%")
        self.reverb_slider.slider.valueChanged.connect(lambda v: self.audio.set_reverb_mix(v / 100.0))
        mod_layout.addWidget(self.reverb_slider)
        tab2_layout.addWidget(mod_group)
        
        lofi_group = QGroupBox("Lo-Fi & Dynamics")
        lofi_layout = QVBoxLayout(lofi_group)
        self.bitcrush_slider = LabeledSlider("Bitcrusher", 0, 100, 0, "%")
        self.bitcrush_slider.slider.valueChanged.connect(lambda v: self.audio.set_bitcrush(v / 100.0))
        lofi_layout.addWidget(self.bitcrush_slider)
        self.sidechain_slider = LabeledSlider("Sidechain Pump", 0, 100, 0, "%")
        self.sidechain_slider.slider.valueChanged.connect(lambda v: self.audio.set_sidechain(v / 100.0))
        lofi_layout.addWidget(self.sidechain_slider)
        tab2_layout.addWidget(lofi_group)
        tab2_layout.addStretch()
        
        # TAB 3: View
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["2D Plane", "3D Waterfall"])
        self.view_combo.currentTextChanged.connect(self.spectrogram.set_view_mode)
        row.addWidget(self.view_combo)
        row.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Inferno", "Viridis", "Plasma", "Spectrum"])
        self.color_combo.currentTextChanged.connect(self.spectrogram.set_colormap)
        row.addWidget(self.color_combo)
        view_layout.addLayout(row)
        self.grid_check = QCheckBox("Show Grid Overlay (2D)")
        self.grid_check.toggled.connect(self.spectrogram.set_grid_overlay)
        view_layout.addWidget(self.grid_check)
        self.brush_slider = LabeledSlider("Brush Size", 1, 5, 2, " px")
        self.brush_slider.slider.valueChanged.connect(self.spectrogram.set_brush_size)
        view_layout.addWidget(self.brush_slider)
        self.yaw_slider = LabeledSlider("3D Yaw", -180, 180, int(np.degrees(-0.5)), "°")
        self.yaw_slider.slider.valueChanged.connect(lambda v: self.spectrogram.set_cam_yaw(np.radians(v)))
        view_layout.addWidget(self.yaw_slider)
        self.pitch_slider = LabeledSlider("3D Pitch", 0, 90, int(np.degrees(0.6)), "°")
        self.pitch_slider.slider.valueChanged.connect(lambda v: self.spectrogram.set_cam_pitch(np.radians(v)))
        view_layout.addWidget(self.pitch_slider)
        tab3_layout.addWidget(view_group)
        tab3_layout.addStretch()
        
        self.tabs.addTab(tab1, "Seq & Synth")
        self.tabs.addTab(tab2, "FX")
        self.tabs.addTab(tab3, "View")
        sidebar_layout.addWidget(self.tabs)
        main_layout.addWidget(sidebar)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready. Try Sprinkling R-Pentominos with Bloom (Gen) rule!")
        self.status_bar.addWidget(self.status_label)
        
        all_sliders = [
            self.sweep_slider, self.speed_slider, self.chaos_slider, self.vol_slider, 
            self.octave_slider, self.blur_slider, self.pluck_slider, self.drive_slider, 
            self.unison_slider, self.fm_slider, self.wt_slider, self.delay_time_slider, 
            self.delay_fb_slider, self.brush_slider, self.yaw_slider, self.pitch_slider, 
            self.lfo_rate_slider, self.lfo_depth_slider, self.reverb_slider, self.attack_slider, 
            self.release_slider, self.sub_amt_slider, self.sub_oct_slider, self.rm_slider,
            self.bitcrush_slider, self.sidechain_slider, self.sync_amt_slider, self.sync_ratio_slider,
            self.pd_slider, self.add_slider, self.chorus_slider, self.formant_slider
        ]
        all_buttons = [self.play_btn, self.random_btn, self.clear_btn, self.drop_btn, self.midi_btn, self.sprinkle_btn]
        all_combos = [self.rule_combo, self.pattern_combo, self.wave_combo, self.scale_combo, self.view_combo, self.color_combo, self.shaper_combo, self.filter_mode_combo, self.vowel_combo, self.rotation_combo]
        
        for w in all_buttons + all_combos + [self.grid_check, self.ping_check] + [s.slider for s in all_sliders]:
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.custom_rule_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step_ca)
        self.update_speed(15)
        
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.sync_ui)
        self.ui_timer.start(33)
        
        self.spectrogram.update_grid(self.ca.grid)
        self.audio.set_grid(self.ca.grid)
        self.spectrogram.notes = self.audio.notes
        self.bars_widget.notes = self.audio.notes

    def update_xy_filter(self, x, y):
        self.audio.set_cutoff(x)
        self.audio.set_resonance(y)

    def on_sweep_changed(self, v):
        secs = v / 10.0
        self.audio.set_sweep(secs)
        self.sweep_slider.value_label.setText(f"{secs:.1f} s")

    def on_rule_changed(self, r):
        self.ca.set_builtin_rule(r)
        self.status_label.setText(f"Rule set to: {r}")

    def on_custom_rule(self):
        rule_str = self.custom_rule_edit.text()
        if self.ca.set_custom_rule(rule_str):
            self.status_label.setText(f"Custom Rule applied: {rule_str}")
            self.rule_combo.setCurrentText("Custom")
        else:
            self.status_label.setText("Invalid Rule. Use format Bxxx/Syyy/Cz (e.g., B3/S23/2)")

    def on_pattern_changed(self, p):
        if p == "None":
            self.spectrogram.armed_pattern = None
            self.spectrogram.setCursor(Qt.CursorShape.ArrowCursor)

    def arm_pattern(self):
        p_name = self.pattern_combo.currentText()
        if p_name == "None":
            self.status_label.setText("Select a pattern to arm first.")
            return
        self.spectrogram.arm_pattern(p_name)
        self.status_label.setText(f"Pattern '{p_name}' armed. Click on the 2D grid to place it.")

    def sprinkle_patterns(self):
        p_name = self.pattern_combo.currentText()
        if p_name == "None":
            self.status_label.setText("Select a pattern to sprinkle first.")
            return
            
        pattern = PATTERNS[p_name]
        max_x = GRID_W - max(p[1] for p in pattern) - 1
        max_y = GRID_H - max(p[0] for p in pattern) - 1
        if max_x <= 0 or max_y <= 0: return
        
        for _ in range(random.randint(5, 10)):
            ox = np.random.randint(0, max_x); oy = np.random.randint(0, max_y)
            for r, c in pattern:
                self.ca.grid[(oy + r) % GRID_H, (ox + c) % GRID_W] = 1.0
        self.status_label.setText(f"Sprinkled {p_name}s across the grid!")

    def place_pattern_at(self, pattern_name, pos):
        pattern = PATTERNS[pattern_name]
        rot_idx = self.rotation_combo.currentIndex()
        
        # Apply Rotation
        if rot_idx == 1: # 90 deg
            pattern = [(c, -r) for r, c in pattern]
        elif rot_idx == 2: # 180 deg
            pattern = [(-r, -c) for r, c in pattern]
        elif rot_idx == 3: # 270 deg
            pattern = [(-c, r) for r, c in pattern]
            
        # Normalize so min row/col is 0
        min_r = min(p[0] for p in pattern)
        min_c = min(p[1] for p in pattern)
        pattern = [(r - min_r, c - min_c) for r, c in pattern]
        
        ox, oy = pos
        for r, c in pattern:
            x, y = ox + c, oy + r
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                self.ca.grid[y, x] = 1.0
        self.status_label.setText("Pattern placed!")

    def export_to_midi(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export to MIDI", "pattern.mid", "MIDI Files (*.mid)")
        if filename:
            try:
                export_midi(filename, self.ca.grid, self.audio.freqs, self.audio.sweep_duration)
                self.status_label.setText(f"Exported MIDI to {filename}")
            except Exception as e:
                self.status_label.setText(f"MIDI Export Error: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space: self.toggle_play()
        elif event.key() == Qt.Key.Key_R: self.randomize()
        elif event.key() == Qt.Key.Key_C: self.clear()
        else: super().keyPressEvent(event)

    def toggle_play(self):
        if self.play_btn.isChecked():
            self.timer.start(); self.play_btn.setText("Pause")
        else:
            self.timer.stop(); self.play_btn.setText("Play")

    def randomize(self):
        self.ca.grid = np.random.choice([0.0, 1.0], size=(GRID_H, GRID_W), p=[0.9, 0.1]).astype(np.float32)

    def clear(self): self.ca.grid[:] = 0

    def update_speed(self, val): self.timer.setInterval(int(1000 / val))

    def step_ca(self): self.ca.step()
        
    def sync_ui(self):
        self.spectrogram.notes = self.audio.notes
        self.spectrogram.update_grid(self.ca.grid)
        self.audio.set_grid(self.ca.grid)
        self.spectrogram.update_playhead(self.audio.playhead_x)
        self.oscilloscope.update_buffer(self.audio.last_buffer)
        self.bars_widget.update_amps(self.audio.current_amps, self.audio.notes)

    def closeEvent(self, event):
        if hasattr(self.audio, 'stream') and self.audio.stream:
            self.audio.stream.stop()
            self.audio.stream.close()
        event.accept()

STYLE_SHEET = """
QWidget { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; }
QPushButton { background-color: #2a2a2a; border: 1px solid #3a3a3a; padding: 6px 12px; border-radius: 4px; color: #e0e0e0; }
QPushButton:hover { background-color: #333333; border: 1px solid #00aaff; }
QPushButton:pressed { background-color: #1a1a1a; }
QPushButton:checked { background-color: #0078cc; color: #ffffff; border: 1px solid #00aaff; }
QComboBox { background-color: #2a2a2a; border: 1px solid #3a3a3a; padding: 4px 8px; border-radius: 4px; color: #e0e0e0; min-width: 70px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #2a2a2a; border: 1px solid #3a3a3a; selection-background-color: #0078cc; }
QSlider::groove:horizontal { border: 1px solid #2a2a2a; height: 4px; background: #1a1a1a; border-radius: 2px; }
QSlider::handle:horizontal { background: #0078cc; border: 1px solid #00aaff; width: 12px; margin: -5px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: #00aaff; }
QGroupBox { border: 1px solid #2a2a2a; border-radius: 6px; margin-top: 14px; padding-top: 10px; font-weight: bold; color: #888; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QCheckBox { spacing: 6px; }
QLabel { color: #b0b0b0; }
QStatusBar { background-color: #1a1a1a; border-top: 1px solid #2a2a2a; color: #888; }
QStatusBar::item { border: none; }
QToolTip { background-color: #2a2a2a; color: #e0e0e0; border: 1px solid #3a3a3a; }
QTabWidget::pane { border: 1px solid #2a2a2a; border-radius: 4px; }
QTabBar::tab { background-color: #1a1a1a; color: #888; padding: 6px 12px; border: 1px solid #2a2a2a; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background-color: #2a2a2a; color: #e0e0e0; border-bottom: 2px solid #00aaff; }
QTabBar::tab:hover { background-color: #222222; }
QLineEdit { background-color: #1a1a1a; border: 1px solid #3a3a3a; padding: 4px; border-radius: 4px; color: #e0e0e0; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())