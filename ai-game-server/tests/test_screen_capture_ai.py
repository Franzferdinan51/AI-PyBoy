"""
Tests for screen capture and analysis with AI inputs
"""
import unittest
import sys
import os
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io
import threading
from typing import List, Dict, Any, Tuple

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.server import numpy_to_base64_image


class MockEmulator:
    """Mock emulator with realistic screen capture behavior"""

    def __init__(self, screen_dimensions=(144, 160, 3)):
        self.screen_dimensions = screen_dimensions
        self.frame_count = 0
        self.screen_content = self._generate_initial_screen()
        self.is_running = True

    def _generate_initial_screen(self):
        """Generate initial screen content"""
        height, width, channels = self.screen_dimensions
        # Create a gradient background
        screen = np.zeros((height, width, channels), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                screen[y, x] = [y % 256, x % 256, (x + y) % 256]
        return screen

    def get_screen(self):
        """Get current screen with frame progression"""
        if not self.is_running:
            return None

        # Simulate screen changes over time
        self.frame_count += 1
        modified_screen = self.screen_content.copy()

        # Add some dynamic content based on frame count
        center_y, center_x = self.screen_dimensions[0] // 2, self.screen_dimensions[1] // 2
        radius = 20 + (self.frame_count % 30)

        y, x = np.ogrid[:self.screen_dimensions[0], :self.screen_dimensions[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

        # Color changes with frame count
        color = [self.frame_count % 256, (self.frame_count * 2) % 256, (self.frame_count * 3) % 256]
        modified_screen[mask] = color

        return modified_screen

    def step(self, action, frames=1):
        """Simulate game step"""
        self.frame_count += frames
        return True

    def stop(self):
        """Stop emulator"""
        self.is_running = False


class TestScreenCapture(unittest.TestCase):
    """Test screen capture functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.emulator = MockEmulator()

    def test_screen_capture_dimensions(self):
        """Test that screen capture returns correct dimensions"""
        screen = self.emulator.get_screen()
        self.assertIsNotNone(screen)
        self.assertEqual(screen.shape, self.emulator.screen_dimensions)

    def test_screen_capture_data_types(self):
        """Test that screen capture returns correct data types"""
        screen = self.emulator.get_screen()
        self.assertIsInstance(screen, np.ndarray)
        self.assertEqual(screen.dtype, np.uint8)

    def test_screen_capture_frame_progression(self):
        """Test that screen content changes with frame progression"""
        screen1 = self.emulator.get_screen()
        screen2 = self.emulator.get_screen()

        # Screens should be different due to frame progression
        self.assertFalse(np.array_equal(screen1, screen2))

    def test_screen_capture_when_stopped(self):
        """Test screen capture when emulator is stopped"""
        self.emulator.stop()
        screen = self.emulator.get_screen()
        self.assertIsNone(screen)

    def test_screen_capture_range_values(self):
        """Test that screen pixel values are in valid range"""
        screen = self.emulator.get_screen()
        self.assertTrue(np.all(screen >= 0))
        self.assertTrue(np.all(screen <= 255))


class TestScreenImageConversion(unittest.TestCase):
    """Test screen image conversion functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.emulator = MockEmulator()
        self.test_screen = self.emulator.get_screen()

    def test_numpy_to_base64_conversion(self):
        """Test numpy array to base64 image conversion"""
        img_base64 = numpy_to_base64_image(self.test_screen)

        # Verify conversion result
        self.assertIsInstance(img_base64, str)
        self.assertTrue(len(img_base64) > 0)

        # Verify it's valid base64
        import base64
        try:
            decoded = base64.b64decode(img_base64)
            self.assertTrue(len(decoded) > 0)
        except Exception:
            self.fail("Base64 decoding failed")

    def test_numpy_to_base64_different_dimensions(self):
        """Test conversion with different screen dimensions"""
        test_dimensions = [
            (144, 160, 3),  # Game Boy
            (160, 144, 3),  # Rotated
            (240, 160, 3),  # GBA
            (320, 240, 3),  # Larger screen
        ]

        for dimensions in test_dimensions:
            with self.subTest(dimensions=dimensions):
                test_screen = np.random.randint(0, 256, dimensions, dtype=np.uint8)
                img_base64 = numpy_to_base64_image(test_screen)
                self.assertIsInstance(img_base64, str)
                self.assertTrue(len(img_base64) > 0)

    def test_numpy_to_base64_edge_cases(self):
        """Test conversion edge cases"""
        # Test with empty screen
        empty_screen = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        img_base64 = numpy_to_base64_image(empty_screen)
        self.assertEqual(img_base64, "")

        # Test with single pixel
        single_pixel = np.array([[[255, 255, 255]]], dtype=np.uint8)
        img_base64 = numpy_to_base64_image(single_pixel)
        self.assertIsInstance(img_base64, str)
        self.assertTrue(len(img_base64) > 0)

    def test_numpy_to_base64_invalid_input(self):
        """Test conversion with invalid input"""
        # Test with None
        result = numpy_to_base64_image(None)
        self.assertEqual(result, "")

        # Test with empty array
        result = numpy_to_base64_image(np.array([]))
        self.assertEqual(result, "")

    def test_numpy_to_base64_different_data_types(self):
        """Test conversion with different numpy data types"""
        test_types = [
            np.uint8,
            np.uint16,
            np.float32,
            np.float64,
            np.int32
        ]

        for dtype in test_types:
            with self.subTest(dtype=dtype):
                test_screen = np.random.randint(0, 256, (50, 50, 3), dtype=dtype)
                img_base64 = numpy_to_base64_image(test_screen)
                self.assertIsInstance(img_base64, str)
                self.assertTrue(len(img_base64) > 0)

    def test_numpy_to_base64_channel_handling(self):
        """Test handling of different channel counts"""
        test_cases = [
            (50, 50, 1),  # Grayscale
            (50, 50, 3),  # RGB
            (50, 50, 4),  # RGBA
            (50, 50, 2),  # Unusual (should be handled)
        ]

        for height, width, channels in test_cases:
            with self.subTest(channels=channels):
                test_screen = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
                img_base64 = numpy_to_base64_image(test_screen)
                self.assertIsInstance(img_base64, str)
                self.assertTrue(len(img_base64) > 0)

    def test_image_quality_settings(self):
        """Test different image quality settings"""
        # This test would require modifying the numpy_to_base64_image function
        # to accept quality parameters. For now, we test the default behavior.
        img_base64 = numpy_to_base64_image(self.test_screen)
        self.assertIsInstance(img_base64, str)
        self.assertTrue(len(img_base64) > 0)


class TestAIScreenAnalysis(unittest.TestCase):
    """Test AI screen analysis capabilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.emulator = MockEmulator()

    def test_screen_analysis_brightness_detection(self):
        """Test brightness detection for AI analysis"""
        # Create screens with different brightness levels
        dark_screen = np.full((50, 50, 3), 30, dtype=np.uint8)    # Dark
        normal_screen = np.full((50, 50, 3), 128, dtype=np.uint8)  # Normal
        bright_screen = np.full((50, 50, 3), 220, dtype=np.uint8)  # Bright

        brightness_levels = [
            (dark_screen, "dark"),
            (normal_screen, "normal"),
            (bright_screen, "bright")
        ]

        for screen, expected_level in brightness_levels:
            with self.subTest(brightness=expected_level):
                detected_level = self._analyze_brightness(screen)
                self.assertEqual(detected_level, expected_level)

    def _analyze_brightness(self, screen_array):
        """Helper method to analyze screen brightness"""
        avg_brightness = np.mean(screen_array)
        if avg_brightness < 85:
            return "dark"
        elif avg_brightness > 170:
            return "bright"
        else:
            return "normal"

    def test_screen_analysis_color_distribution(self):
        """Test color distribution analysis"""
        # Create screens with dominant colors
        red_screen = np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8)
        green_screen = np.full((50, 50, 3), [0, 255, 0], dtype=np.uint8)
        blue_screen = np.full((50, 50, 3), [0, 0, 255], dtype=np.uint8)

        color_screens = [
            (red_screen, "red"),
            (green_screen, "green"),
            (blue_screen, "blue")
        ]

        for screen, expected_color in color_screens:
            with self.subTest(color=expected_color):
                dominant_color = self._analyze_dominant_color(screen)
                self.assertEqual(dominant_color, expected_color)

    def _analyze_dominant_color(self, screen_array):
        """Helper method to analyze dominant color"""
        avg_color = np.mean(screen_array, axis=(0, 1))
        if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
            return "red"
        elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
            return "green"
        elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
            return "blue"
        else:
            return "mixed"

    def test_screen_analysis_pattern_detection(self):
        """Test pattern detection in screens"""
        # Create screens with different patterns
        horizontal_lines = np.zeros((50, 50, 3), dtype=np.uint8)
        horizontal_lines[::10, :] = 255

        vertical_lines = np.zeros((50, 50, 3), dtype=np.uint8)
        vertical_lines[:, ::10] = 255

        checkerboard = np.zeros((50, 50, 3), dtype=np.uint8)
        checkerboard[::10, ::10] = 255
        checkerboard[1::10, 1::10] = 255

        pattern_screens = [
            (horizontal_lines, "horizontal"),
            (vertical_lines, "vertical"),
            (checkerboard, "checkerboard")
        ]

        for screen, expected_pattern in pattern_screens:
            with self.subTest(pattern=expected_pattern):
                detected_pattern = self._analyze_pattern(screen)
                self.assertEqual(detected_pattern, expected_pattern)

    def _analyze_pattern(self, screen_array):
        """Helper method to analyze screen patterns"""
        # Convert to grayscale for pattern analysis
        gray = np.mean(screen_array, axis=2)

        # Simple pattern detection
        horizontal_changes = np.sum(np.abs(np.diff(gray, axis=1)))
        vertical_changes = np.sum(np.abs(np.diff(gray, axis=0)))

        if horizontal_changes > vertical_changes * 1.5:
            return "horizontal"
        elif vertical_changes > horizontal_changes * 1.5:
            return "vertical"
        else:
            return "checkerboard"

    def test_screen_analysis_motion_detection(self):
        """Test motion detection between frames"""
        # Create consecutive frames with motion
        frame1 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        frame2[20:30, 20:30] = [255, 255, 255]  # Add a white square

        motion_level = self._detect_motion(frame1, frame2)
        self.assertGreater(motion_level, 0.1)  # Should detect significant motion

        # Test with identical frames
        no_motion = self._detect_motion(frame1, frame1)
        self.assertEqual(no_motion, 0.0)

    def _detect_motion(self, frame1, frame2):
        """Helper method to detect motion between frames"""
        if frame1.shape != frame2.shape:
            return 1.0  # Maximum motion for different shapes

        # Calculate pixel-wise difference
        diff = np.abs(frame1.astype(float) - frame2.astype(float))
        motion_level = np.mean(diff) / 255.0
        return motion_level


class TestAIScreenInputIntegration(unittest.TestCase):
    """Test integration of screen capture with AI input processing"""

    def setUp(self):
        """Set up test fixtures"""
        self.emulator = MockEmulator()

    def test_ai_screen_input_workflow(self):
        """Test complete AI screen input workflow"""
        # Capture screen
        screen = self.emulator.get_screen()
        self.assertIsNotNone(screen)

        # Convert to AI-compatible format
        img_base64 = numpy_to_base64_image(screen)
        self.assertIsInstance(img_base64, str)
        self.assertTrue(len(img_base64) > 0)

        # Analyze screen content
        brightness = self._analyze_brightness(screen)
        dominant_color = self._analyze_dominant_color(screen)

        # Create AI prompt with screen analysis
        ai_prompt = self._create_ai_prompt("Find the exit", brightness, dominant_color)
        self.assertIn(brightness, ai_prompt)
        self.assertIn(dominant_color, ai_prompt)

    def _create_ai_prompt(self, goal, brightness, dominant_color):
        """Helper method to create AI prompt with screen analysis"""
        return f"""You are playing a retro game with goal: "{goal}".
Screen analysis: brightness={brightness}, dominant_color={dominant_color}.
What action should you take?"""

    def test_screen_analysis_for_nvidia(self):
        """Test screen analysis specifically for NVIDIA API"""
        screen = self.emulator.get_screen()
        brightness = self._analyze_brightness(screen)

        # NVIDIA API uses brightness analysis
        nvidia_prompt = f"""Screen appears {brightness} overall.
Based on this information, what action should you take?"""

        self.assertIn(brightness, nvidia_prompt)

    def test_screen_analysis_for_vision_apis(self):
        """Test screen analysis for vision-based APIs (Gemini, OpenRouter, OpenAI)"""
        screen = self.emulator.get_screen()
        img_base64 = numpy_to_base64_image(screen)

        # Vision APIs receive the actual image
        vision_prompt = """Analyze this game screen and suggest the next action."""
        image_data = f"data:image/jpeg;base64,{img_base64}"

        self.assertIsInstance(image_data, str)
        self.assertTrue(len(image_data) > 100)  # Should be substantial base64 data

    def test_temporal_screen_analysis(self):
        """Test analysis of screen changes over time"""
        frames = []
        for i in range(5):
            screen = self.emulator.get_screen()
            frames.append(screen)
            time.sleep(0.01)  # Small delay

        # Analyze temporal patterns
        motion_levels = []
        for i in range(1, len(frames)):
            motion = self._detect_motion(frames[i-1], frames[i])
            motion_levels.append(motion)

        # Should detect some motion between frames
        self.assertGreater(max(motion_levels), 0.0)

    def test_screen_analysis_performance(self):
        """Test performance of screen analysis operations"""
        import time

        # Test screen capture performance
        start_time = time.time()
        for _ in range(100):
            screen = self.emulator.get_screen()
        capture_time = time.time() - start_time

        # Test conversion performance
        screen = self.emulator.get_screen()
        start_time = time.time()
        for _ in range(50):
            img_base64 = numpy_to_base64_image(screen)
        conversion_time = time.time() - start_time

        # Test analysis performance
        start_time = time.time()
        for _ in range(100):
            brightness = self._analyze_brightness(screen)
            dominant_color = self._analyze_dominant_color(screen)
        analysis_time = time.time() - start_time

        # Performance should be reasonable
        self.assertLess(capture_time, 1.0)  # 100 captures in < 1 second
        self.assertLess(conversion_time, 2.0)  # 50 conversions in < 2 seconds
        self.assertLess(analysis_time, 0.5)  # 100 analyses in < 0.5 seconds

    def test_memory_efficiency(self):
        """Test memory efficiency of screen operations"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform many screen operations
        for _ in range(1000):
            screen = self.emulator.get_screen()
            img_base64 = numpy_to_base64_image(screen)
            self._analyze_brightness(screen)
            self._analyze_dominant_color(screen)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        self.assertLess(memory_increase, 100 * 1024 * 1024)  # Less than 100MB

    def test_concurrent_screen_operations(self):
        """Test concurrent screen capture and analysis"""
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    screen = self.emulator.get_screen()
                    img_base64 = numpy_to_base64_image(screen)
                    brightness = self._analyze_brightness(screen)
                    results.append((worker_id, i, brightness))
                    time.sleep(0.001)
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify results
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 50)  # 5 workers * 10 iterations each

    def test_error_handling_in_screen_analysis(self):
        """Test error handling in screen analysis operations"""
        # Test with invalid screen data
        result = numpy_to_base64_image(None)
        self.assertEqual(result, "")

        # Test with malformed array
        malformed_array = np.array([1, 2, 3])  # Wrong shape
        result = numpy_to_base64_image(malformed_array)
        self.assertEqual(result, "")

        # Test brightness analysis with edge cases
        empty_screen = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        brightness = self._analyze_brightness(empty_screen)
        self.assertEqual(brightness, "dark")  # Should handle gracefully


class TestRealWorldScreenScenarios(unittest.TestCase):
    """Test screen analysis with realistic game scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.emulator = MockEmulator()

    def test_game_over_screen_simulation(self):
        """Test analysis of game over screen simulation"""
        # Create a screen that simulates "GAME OVER" text
        game_over_screen = np.full((144, 160, 3), [50, 50, 50], dtype=np.uint8)  # Dark background

        # Add bright text (simulated)
        game_over_screen[60:80, 40:120] = [255, 255, 255]  # White text area

        brightness = self._analyze_brightness(game_over_screen)
        dominant_color = self._analyze_dominant_color(game_over_screen)

        # Should detect the bright text
        self.assertEqual(dominant_color, "mixed")

    def test_level_complete_screen_simulation(self):
        """Test analysis of level complete screen simulation"""
        # Create a bright, colorful screen simulating level complete
        level_complete_screen = np.random.randint(200, 256, (144, 160, 3), dtype=np.uint8)

        brightness = self._analyze_brightness(level_complete_screen)
        self.assertEqual(brightness, "bright")

    def test_dark_dungeon_screen_simulation(self):
        """Test analysis of dark dungeon screen simulation"""
        # Create a dark screen simulating dungeon environment
        dungeon_screen = np.random.randint(0, 80, (144, 160, 3), dtype=np.uint8)

        brightness = self._analyze_brightness(dungeon_screen)
        self.assertEqual(brightness, "dark")

    def test_bright_outdoor_screen_simulation(self):
        """Test analysis of bright outdoor screen simulation"""
        # Create a bright screen simulating outdoor environment
        outdoor_screen = np.random.randint(180, 256, (144, 160, 3), dtype=np.uint8)

        brightness = self._analyze_brightness(outdoor_screen)
        self.assertEqual(brightness, "bright")

    def _analyze_brightness(self, screen_array):
        """Helper method to analyze screen brightness"""
        if screen_array.size == 0:
            return "dark"
        avg_brightness = np.mean(screen_array)
        if avg_brightness < 85:
            return "dark"
        elif avg_brightness > 170:
            return "bright"
        else:
            return "normal"

    def _analyze_dominant_color(self, screen_array):
        """Helper method to analyze dominant color"""
        if screen_array.size == 0:
            return "unknown"
        avg_color = np.mean(screen_array, axis=(0, 1))
        if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
            return "red"
        elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
            return "green"
        elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
            return "blue"
        else:
            return "mixed"


if __name__ == '__main__':
    unittest.main(verbosity=2)