"""
QuantumFlow AI - Model Inference Engine
Optimized for real-time trading decisions
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TradingInferenceEngine:
    """
    Optimized inference engine for live trading

    Features:
    - Model quantization for faster inference
    - Batch prediction support
    - Latency monitoring
    - Model versioning
    - A/B testing support
    """

    def __init__(self, model_path: str, device: str = 'cpu', use_quantization: bool = False):
        self.device = device
        self.model_path = model_path
        self.use_quantization = use_quantization

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Performance tracking
        self.inference_times = []
        self.total_predictions = 0

        # Warm up
        self._warmup()

        logger.info(f"🧠 Inference Engine loaded: {model_path}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Quantization: {use_quantization}")

    def _load_model(self):
        """Load and optionally quantize model"""
        from agents.policy_network import TransformerPolicyNetwork, EnsemblePolicy

        # Try to load as ensemble first
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            if 'ensemble' in self.model_path or 'ensemble' in str(checkpoint.keys()):
                model = EnsemblePolicy(
                    n_features=256,
                    window_size=128,
                    ensemble_size=3,
                )
            else:
                model = TransformerPolicyNetwork(
                    n_features=256,
                    window_size=128,
                )

            model.load_state_dict(checkpoint['policy_state_dict'])

            # Quantize if requested
            if self.use_quantization and self.device == 'cpu':
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                logger.info("✅ Model quantized for CPU inference")

            return model.to(self.device)

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def _warmup(self, num_warmup: int = 10):
        """Warm up model with dummy inputs"""
        dummy_input = torch.randn(1, 128, 256).to(self.device)

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(dummy_input)

        logger.info("✅ Model warmed up")

    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Make trading decision

        Args:
            observation: (window, features) or (batch, window, features)
            deterministic: If True, use mean action (no sampling)

        Returns:
            action: Trading action [direction, size, sl, tp]
            info: Dict with confidence, latency, etc.
        """
        start_time = time.time()

        # Prepare input
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)

        obs_tensor = torch.FloatTensor(observation).to(self.device)

        # Inference
        with torch.no_grad():
            if isinstance(self.model, EnsemblePolicy):
                action, value, info = self.model.get_action(obs_tensor, deterministic=deterministic)

                # Extract ensemble info
                ensemble_info = {
                    'individual_actions': info.get('individual_actions'),
                    'weights': info.get('weights'),
                    'agent_names': info.get('agent_names'),
                }
            else:
                action, value = self.model.get_action(obs_tensor, deterministic=deterministic)
                ensemble_info = {}

        inference_time = (time.time() - start_time) * 1000  # ms

        # Track performance
        self.inference_times.append(inference_time)
        self.total_predictions += 1

        # Build info
        info = {
            'latency_ms': inference_time,
            'value': float(value.cpu().numpy()[0][0]) if value is not None else 0,
            'confidence': self._calculate_confidence(action),
            **ensemble_info,
        }

        return action, info

    def predict_batch(self, observations: list) -> list:
        """Batch prediction for multiple observations"""
        batch = np.stack(observations)
        obs_tensor = torch.FloatTensor(batch).to(self.device)

        with torch.no_grad():
            if isinstance(self.model, EnsemblePolicy):
                actions, values, _ = self.model(obs_tensor)
            else:
                actions, values, _ = self.model(obs_tensor)

        return actions.cpu().numpy()

    def _calculate_confidence(self, action: np.ndarray) -> float:
        """Calculate confidence score for action"""
        # Higher confidence when action is more decisive
        direction = abs(action[0][0]) if action.ndim > 1 else abs(action[0])
        size = action[0][1] if action.ndim > 1 else action[1]

        # Confidence based on direction strength and position size
        confidence = (direction * 0.5 + size * 0.5)
        return float(np.clip(confidence, 0, 1))

    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {}

        times = np.array(self.inference_times)

        return {
            'total_predictions': self.total_predictions,
            'avg_latency_ms': float(np.mean(times)),
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'p95_latency_ms': float(np.percentile(times, 95)),
            'p99_latency_ms': float(np.percentile(times, 99)),
        }

    def export_onnx(self, output_path: str):
        """Export model to ONNX format for faster inference"""
        dummy_input = torch.randn(1, 128, 256).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action', 'value'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'action': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            }
        )

        logger.info(f"✅ Model exported to ONNX: {output_path}")

class ModelVersionManager:
    """Manage multiple model versions for A/B testing"""

    def __init__(self):
        self.models = {}
        self.active_model = None
        self.performance_log = {}

    def register_model(self, name: str, model_path: str, weight: float = 1.0):
        """Register a model version"""
        self.models[name] = {
            'path': model_path,
            'weight': weight,
            'engine': None,
        }
        logger.info(f"✅ Registered model: {name} (weight: {weight})")

    def load_model(self, name: str):
        """Load a specific model"""
        if name not in self.models:
            logger.error(f"❌ Model {name} not found")
            return False

        self.models[name]['engine'] = TradingInferenceEngine(
            self.models[name]['path']
        )
        self.active_model = name
        logger.info(f"✅ Loaded model: {name}")
        return True

    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Predict using active model"""
        if not self.active_model:
            raise ValueError("No active model")

        engine = self.models[self.active_model]['engine']
        return engine.predict(observation)

    def ensemble_predict(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Weighted ensemble prediction across all models"""
        actions = []
        weights = []

        for name, model_info in self.models.items():
            if model_info['engine']:
                action, _ = model_info['engine'].predict(observation)
                actions.append(action)
                weights.append(model_info['weight'])

        if not actions:
            raise ValueError("No models loaded")

        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_action = sum(a * w for a, w in zip(actions, weights))

        return ensemble_action, {
            'models_used': list(self.models.keys()),
            'weights': weights.tolist(),
        }
