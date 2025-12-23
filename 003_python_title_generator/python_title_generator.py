#!/usr/bin/env python3
"""
Title Video Generator
Creates a video with typewriter effect, blinking cursor, and very subtle typewriter sound effects.

Usage:
    python generate_title_video.py "Your text here" output.mp4
    
    - Text: The text to display
    - Output: Output file (e.g. output.mp4)
    
Features:
    - Typewriter effect with character-by-character reveal
    - Randomized delays (40-200ms) for natural typing rhythm
    - Blinking block cursor
    - Automatic text wrapping with proper margins
    - Very subtle typewriter sound for each character
    - Full HD (1920x1080) output
"""

import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication, QLabel, QWidget
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QFont, QPainter, QPixmap, QColor, QFontMetrics
import subprocess
import tempfile
import shutil
import numpy as np
import wave
import random


class TitleWidget(QWidget):
    """Widget for displaying animated title text"""
    
    def __init__(self, text, output_path):
        super().__init__()
        self.full_text = text
        self.output_path = output_path
        
        # Animation state
        self.current_text = ""
        self.current_index = 0
        self.cursor_visible = True
        self.animation_finished = False
        self.end_wait_frames = 0
        
        # Random delay settings (40-200ms between characters)
        self.min_delay_ms = 40
        self.max_delay_ms = 200
        self.next_char_delay = random.randint(self.min_delay_ms, self.max_delay_ms)
        
        # Frame capture
        self.frames = []
        self.temp_dir = tempfile.mkdtemp()
        self.frame_number = 0
        
        # FPS for the video (30 fps)
        self.fps = 30
        self.frame_time_ms = 1000 // self.fps  # approx. 33ms per frame
        
        # Audio setup
        self.sample_rate = 44100  # Hz
        self.audio_timeline = []
        self.generate_typewriter_sound()
        
        self.init_ui()
        self.start_animation()
    
    def generate_typewriter_sound(self):
        """Generate a very subtle typewriter click sound"""
        duration = 0.035  # 35ms sound (even shorter)
        samples = int(self.sample_rate * duration)
        
        # Create a very subtle click sound with quick attack and decay
        t = np.linspace(0, duration, samples)
        
        # Mix of frequencies for realistic but very subtle typewriter sound
        freq1 = 800  # Main click frequency
        freq2 = 1200  # Harmonic
        freq3 = 400   # Bass component
        
        # Much lower amplitudes for very quiet sound
        sound = (
            0.04 * np.sin(2 * np.pi * freq1 * t) +
            0.025 * np.sin(2 * np.pi * freq2 * t) +
            0.015 * np.sin(2 * np.pi * freq3 * t)
        )
        
        # Apply envelope (quick attack, exponential decay)
        envelope = np.exp(-t * 70)  # Even faster decay for more subtle sound
        sound = sound * envelope
        
        # Add minimal noise
        noise = np.random.normal(0, 0.002, samples)
        sound = sound + noise
        
        # Normalize to prevent clipping, but keep it very quiet
        sound = sound / np.max(np.abs(sound)) * 0.08  # Much lower volume (was 0.15)
        
        self.typewriter_sound = sound
        
    def init_ui(self):
        """Initialize UI"""
        # Window size
        self.setFixedSize(1920, 1080)
        self.setWindowTitle("Title Video Generator")
        
        # Black background
        self.setStyleSheet("background-color: black;")
        
        # Label for the text
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(0, 0, 1920, 1080)
        self.label.setWordWrap(True)
        
        # Font
        font = QFont("Courier New", 72, QFont.Bold)
        self.label.setFont(font)
        self.label.setStyleSheet("color: white; background-color: black;")
        
        # Calculate text area with margins (10% margin on each side)
        self.text_width = int(1920 * 0.8)  # 80% of screen width
        self.margin_left = int(1920 * 0.1)  # 10% margin
        
    def wrap_text(self, text):
        """Wrap text into lines that fit the available width"""
        metrics = QFontMetrics(self.label.font())
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            # Try adding the word to the current line
            test_line = ' '.join(current_line + [word])
            line_width = metrics.horizontalAdvance(test_line + "█")  # Include cursor width
            
            if line_width <= self.text_width:
                current_line.append(word)
            else:
                # Current line is full, start a new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                
                # Check if single word is too long
                if metrics.horizontalAdvance(word + "█") > self.text_width:
                    # Word itself is too long, we need to break it
                    # For now, just add it anyway (could be improved)
                    pass
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
        
    def start_animation(self):
        """Start animation"""
        # Timer for frame updates
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(self.frame_time_ms)
        
        # Counter for character timing
        self.char_timer_counter = 0
        self.cursor_blink_counter = 0
        
    def update_frame(self):
        """Frame update for animation"""
        if not self.animation_finished:
            # Add character with random delay
            self.char_timer_counter += self.frame_time_ms
            
            if self.char_timer_counter >= self.next_char_delay and self.current_index < len(self.full_text):
                self.current_text += self.full_text[self.current_index]
                self.current_index += 1
                self.char_timer_counter = 0
                
                # Generate new random delay for next character (50-150ms)
                self.next_char_delay = random.randint(self.min_delay_ms, self.max_delay_ms)
                
                # Add typewriter sound to audio timeline
                # Calculate timestamp in seconds
                timestamp = (self.frame_number / self.fps)
                self.audio_timeline.append(timestamp)
            
            # Cursor blink (every 500ms)
            self.cursor_blink_counter += self.frame_time_ms
            if self.cursor_blink_counter >= 500:
                self.cursor_visible = not self.cursor_visible
                self.cursor_blink_counter = 0
            
            # Wrap text to multiple lines
            wrapped_text = self.wrap_text(self.current_text)
            
            # Display text with cursor
            if self.current_index < len(self.full_text):
                # During typing: cursor always visible
                display_text = wrapped_text + "█"
            else:
                # After typing: cursor blinks
                if self.cursor_visible:
                    display_text = wrapped_text + "█"
                else:
                    display_text = wrapped_text + " "
            
            self.label.setText(display_text)
            
            # Check if animation finished
            if self.current_index >= len(self.full_text):
                self.end_wait_frames += 1
                # Wait 3 seconds = 3 * fps frames
                if self.end_wait_frames >= 3 * self.fps:
                    self.animation_finished = True
        else:
            # Animation finished - create video
            self.frame_timer.stop()
            self.create_audio()
            self.create_video()
            QApplication.quit()
            return
        
        # Save frame
        self.capture_frame()
    
    def capture_frame(self):
        """Save current frame as image"""
        # Screenshot of widget
        pixmap = self.grab()
        
        # Save frame
        frame_path = os.path.join(self.temp_dir, f"frame_{self.frame_number:06d}.png")
        pixmap.save(frame_path)
        self.frames.append(frame_path)
        self.frame_number += 1
    
    def create_audio(self):
        """Create audio file from timeline"""
        print(f"\nCreating audio with {len(self.audio_timeline)} typewriter sounds...")
        
        # Calculate total duration (video duration)
        video_duration = self.frame_number / self.fps
        total_samples = int(video_duration * self.sample_rate)
        
        # Create empty audio buffer
        audio_buffer = np.zeros(total_samples)
        
        # Add each typewriter sound at its timestamp
        for timestamp in self.audio_timeline:
            start_sample = int(timestamp * self.sample_rate)
            end_sample = start_sample + len(self.typewriter_sound)
            
            # Make sure we don't exceed buffer length
            if end_sample <= total_samples:
                audio_buffer[start_sample:end_sample] += self.typewriter_sound
            else:
                # Truncate sound if it goes beyond buffer
                remaining = total_samples - start_sample
                if remaining > 0:
                    audio_buffer[start_sample:] += self.typewriter_sound[:remaining]
        
        # Normalize to prevent clipping but keep it very subtle
        max_val = np.max(np.abs(audio_buffer))
        if max_val > 0:
            audio_buffer = audio_buffer / max_val * 0.15  # Very low volume (was 0.3)
        
        # Convert to 16-bit PCM
        audio_data = (audio_buffer * 32767).astype(np.int16)
        
        # Save as WAV file
        self.audio_path = os.path.join(self.temp_dir, "audio.wav")
        with wave.open(self.audio_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        print(f"✓ Audio created: {self.audio_path}")
    
    def create_video(self):
        """Create video from frames"""
        print(f"\nCreating video with {len(self.frames)} frames...")
        print(f"FPS: {self.fps}")
        
        # ffmpeg command with audio
        frame_pattern = os.path.join(self.temp_dir, "frame_%06d.png")
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite without asking
            '-framerate', str(self.fps),
            '-i', frame_pattern,
            '-i', self.audio_path,  # Add audio input
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',  # Audio codec
            '-b:a', '192k',  # Audio bitrate
            '-shortest',  # End when shortest stream ends
            self.output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"\n✓ Video successfully created: {self.output_path}")
            print(f"✓ Video includes typewriter sound effects")
            
            # Cleanup
            shutil.rmtree(self.temp_dir)
            print(f"✓ Temporary files deleted")
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error creating video:")
            print(e.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("\n✗ Error: ffmpeg not found!")
            print("Please install ffmpeg:")
            print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("  macOS: brew install ffmpeg")
            print("  Windows: https://ffmpeg.org/download.html")
            sys.exit(1)


def main():
    """Main function"""
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python generate_title_video.py \"Text\" <output.mp4>")
        print("\nExample:")
        print('  python generate_title_video.py "Once upon a time ..." output.mp4')
        print("\nParameters:")
        print("  Text: The text to display (in quotes)")
        print("  output.mp4: Name of the output file")
        print("\nNote: Character delays are randomized between 40-200ms for natural typing effect")
        sys.exit(1)
    
    text = sys.argv[1]
    output_path = sys.argv[2]
    
    # Check if output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        print(f"✗ Error: Directory does not exist: {output_dir}")
        sys.exit(1)
    
    # Calculate approximate duration (using average delay of 120ms)
    avg_delay_ms = 120
    approx_duration = (len(text) * avg_delay_ms / 1000) + 3
    
    # Print info
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          Title Video Generator                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nText: \"{text}\"")
    print(f"Character delay: Random (40-200ms)")
    print(f"Output: {output_path}")
    print(f"\nApprox. duration: {approx_duration:.1f} seconds")
    print("\nStarting rendering...")
    
    # Start Qt application
    app = QApplication(sys.argv)
    
    # Create widget (hidden as we're only rendering)
    widget = TitleWidget(text, output_path)
    widget.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()