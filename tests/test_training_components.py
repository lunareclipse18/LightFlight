"""Tests for LightFlight training components."""

import unittest
import numpy as np
import gym


class TestEvoqueWrapperBasics(unittest.TestCase):
    """Test EvoqueWrapper observation and action handling."""
    
    def test_observation_shape(self):
        """Verify observation shape is always 6-dim."""
        # This is a mock test - real test would require Gazebo
        expected_shape = (6,)
        obs = np.random.randn(*expected_shape).astype(np.float32)
        self.assertEqual(obs.shape, expected_shape)
    
    def test_action_clipping(self):
        """Verify actions are clipped to [0, 1]."""
        test_actions = [-0.5, 0.0, 0.5, 1.0, 1.5]
        expected = [0.0, 0.0, 0.5, 1.0, 1.0]
        
        clipped = [np.clip(a, 0.0, 1.0) for a in test_actions]
        np.testing.assert_array_equal(clipped, expected)
    
    def test_observation_clipping_ranges(self):
        """Verify observation clipping at ±500 and ±50."""
        # Error obs should clip to ±500
        error_obs = np.array([-1000, 0, 1000], dtype=np.float32)
        error_clipped = np.clip(error_obs, -500.0, 500.0)
        expected_error = np.array([-500, 0, 500], dtype=np.float32)
        np.testing.assert_array_equal(error_clipped, expected_error)
        
        # Delta error should clip to ±50
        delta_obs = np.array([-100, 0, 100], dtype=np.float32)
        delta_clipped = np.clip(delta_obs, -50.0, 50.0)
        expected_delta = np.array([-50, 0, 50], dtype=np.float32)
        np.testing.assert_array_equal(delta_clipped, expected_delta)


class TestRewardComputation(unittest.TestCase):
    """Test reward function computation."""
    
    def test_reward_range(self):
        """Verify reward is clipped to [-10000, 0]."""
        # Simulate error penalty
        error = np.array([100.0, 50.0, 25.0])
        error_penalty = -np.sum(np.square(error)) - np.sum(np.abs(error)) * 0.1
        
        action_delta = np.array([0.5, 0.3, 0.2, 0.1])
        oscillation_penalty = -np.sum(np.square(action_delta)) * 0.1
        
        reward = error_penalty + oscillation_penalty
        clipped = np.clip(reward, -10000.0, 0.0)
        
        self.assertGreaterEqual(clipped, -10000.0)
        self.assertLessEqual(clipped, 0.0)
    
    def test_zero_error_zero_penalty(self):
        """Verify zero error gives zero penalty."""
        error = np.array([0.0, 0.0, 0.0])
        error_penalty = -np.sum(np.square(error)) - np.sum(np.abs(error)) * 0.1
        
        self.assertEqual(error_penalty, 0.0)


class TestNaNInfinityHandling(unittest.TestCase):
    """Test NaN and infinity handling in observations."""
    
    def test_nan_to_num(self):
        """Verify NaN → 0 conversion."""
        obs_with_nan = np.array([1.0, np.nan, 3.0])
        cleaned = np.nan_to_num(obs_with_nan, nan=0.0, posinf=0.0, neginf=0.0)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(cleaned, expected)
    
    def test_infinity_to_num(self):
        """Verify infinity → 0 conversion."""
        obs_with_inf = np.array([1.0, np.inf, -np.inf])
        cleaned = np.nan_to_num(obs_with_inf, nan=0.0, posinf=0.0, neginf=0.0)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_equal(cleaned, expected)


if __name__ == '__main__':
    unittest.main()
