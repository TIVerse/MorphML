"""Learning curve extrapolation for early stopping.

Predicts final performance from early training epochs.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import List, Optional, Tuple

import numpy as np

from morphml.logging_config import get_logger

logger = get_logger(__name__)

try:
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class LearningCurvePredictor:
    """
    Predict final accuracy from early training epochs.

    Uses curve fitting to extrapolate learning curves:
    - Power law: acc(t) = a - b * t^(-c)
    - Exponential: acc(t) = a * (1 - exp(-b * t))

    Example:
        >>> predictor = LearningCurvePredictor()
        >>>
        >>> # Observe first 10 epochs
        >>> epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> accuracies = [0.3, 0.45, 0.55, 0.62, 0.67, 0.71, 0.74, 0.76, 0.78, 0.79]
        >>>
        >>> # Predict final accuracy at epoch 200
        >>> final_acc = predictor.predict_final_accuracy(accuracies, epochs, final_epoch=200)
        >>> print(f"Predicted final: {final_acc:.3f}")
    """

    def __init__(self, curve_type: str = "power_law"):
        """
        Initialize predictor.

        Args:
            curve_type: Type of curve to fit ('power_law' or 'exponential')
        """
        self.curve_type = curve_type
        logger.info(f"Initialized LearningCurvePredictor (type={curve_type})")

    @staticmethod
    def power_law(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Power law curve: acc(t) = a - b * t^(-c)

        Args:
            t: Time steps (epochs)
            a: Asymptotic accuracy
            b: Scale parameter
            c: Decay rate

        Returns:
            Predicted accuracies
        """
        return a - b * np.power(t, -c)

    @staticmethod
    def exponential(t: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Exponential curve: acc(t) = a * (1 - exp(-b * t))

        Args:
            t: Time steps (epochs)
            a: Asymptotic accuracy
            b: Convergence rate

        Returns:
            Predicted accuracies
        """
        return a * (1 - np.exp(-b * t))

    def predict_final_accuracy(
        self,
        observed_accuracies: List[float],
        observed_epochs: Optional[List[int]] = None,
        final_epoch: int = 200,
    ) -> float:
        """
        Extrapolate learning curve to predict final accuracy.

        Args:
            observed_accuracies: Observed accuracies
            observed_epochs: Corresponding epochs (default: [1, 2, 3, ...])
            final_epoch: Epoch to predict

        Returns:
            Predicted final accuracy
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, returning last observed accuracy")
            return observed_accuracies[-1] if observed_accuracies else 0.5

        # Default epochs
        if observed_epochs is None:
            observed_epochs = list(range(1, len(observed_accuracies) + 1))

        if len(observed_accuracies) < 3:
            logger.warning("Too few observations for curve fitting")
            return observed_accuracies[-1] if observed_accuracies else 0.5

        # Convert to arrays
        t = np.array(observed_epochs, dtype=float)
        acc = np.array(observed_accuracies, dtype=float)

        try:
            if self.curve_type == "power_law":
                # Fit power law
                params, _ = curve_fit(
                    self.power_law,
                    t,
                    acc,
                    p0=[0.9, 0.1, 0.5],
                    bounds=([0, 0, 0], [1.0, 1.0, 5.0]),
                    maxfev=1000,
                )

                # Predict
                prediction = self.power_law(final_epoch, *params)

            elif self.curve_type == "exponential":
                # Fit exponential
                params, _ = curve_fit(
                    self.exponential,
                    t,
                    acc,
                    p0=[0.9, 0.01],
                    bounds=([0, 0], [1.0, 1.0]),
                    maxfev=1000,
                )

                # Predict
                prediction = self.exponential(final_epoch, *params)

            else:
                raise ValueError(f"Unknown curve type: {self.curve_type}")

            # Clip to valid range
            prediction = np.clip(prediction, 0.0, 1.0)

            logger.debug(
                f"Extrapolated from {len(acc)} epochs to epoch {final_epoch}: " f"{prediction:.4f}"
            )

            return float(prediction)

        except Exception as e:
            logger.warning(f"Curve fitting failed: {e}, using last observed")
            return observed_accuracies[-1]

    def should_early_stop(
        self,
        observed_accuracies: List[float],
        observed_epochs: Optional[List[int]] = None,
        threshold: float = 0.8,
        confidence: float = 0.95,
    ) -> bool:
        """
        Decide whether to stop training early.

        Args:
            observed_accuracies: Observed accuracies
            observed_epochs: Corresponding epochs
            threshold: Minimum required final accuracy
            confidence: Confidence level for prediction

        Returns:
            True if should stop early (predicted to not reach threshold)
        """
        # Predict final accuracy
        predicted_final = self.predict_final_accuracy(observed_accuracies, observed_epochs)

        # Conservative: add margin for uncertainty
        margin = 0.05  # 5% margin
        predicted_with_margin = predicted_final - margin

        # Stop if predicted final is below threshold
        should_stop = predicted_with_margin < threshold

        if should_stop:
            logger.info(
                f"Early stopping recommended: predicted final {predicted_final:.3f} "
                f"(with margin: {predicted_with_margin:.3f}) < threshold {threshold:.3f}"
            )

        return should_stop

    def fit_curve(
        self,
        observed_accuracies: List[float],
        observed_epochs: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit curve to observed data and return full trajectory.

        Args:
            observed_accuracies: Observed accuracies
            observed_epochs: Corresponding epochs

        Returns:
            (fitted_epochs, fitted_accuracies) for plotting
        """
        if observed_epochs is None:
            observed_epochs = list(range(1, len(observed_accuracies) + 1))

        # Generate dense time steps for smooth curve
        max_epoch = max(observed_epochs)
        fitted_epochs = np.linspace(1, max_epoch * 2, 100)

        # Predict at each point
        fitted_accuracies = []
        for epoch in fitted_epochs:
            pred = self.predict_final_accuracy(
                observed_accuracies, observed_epochs, final_epoch=int(epoch)
            )
            fitted_accuracies.append(pred)

        return fitted_epochs, np.array(fitted_accuracies)
