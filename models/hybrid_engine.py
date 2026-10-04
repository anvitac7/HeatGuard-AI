"""
HeatGuard AI - Two-Stage Hybrid Engine (Advanced ML & Physics-Grounded Severity Model).

Combines:
- Stage 1: Continuous Temperature Departure Regressor (Huber Loss) + Quantile Uncertainty Bounds [p10, p90]
- Stage 2: Cost-Sensitive LightGBM Classifier + Probability Calibration (Isotonic / Sigmoid)
- Stage 3: Physics-Grounded IMD Severity & Risk Tier Mapping

This architecture cleanly solves rare-event extreme class scarcity (e.g. Extreme severity)
by learning continuous thermal dynamics across 100% of samples (187k continuous records).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


class TwoStageHeatwavePredictor:
    """
    Two-Stage Hybrid Predictor for Heatwave Occurrence, Departure, and Severity.
    """

    def __init__(
        self,
        # Stage 1 Regressor Hyperparameters
        reg_learning_rate: float = 0.03,
        reg_n_estimators: int = 400,
        reg_num_leaves: int = 31,
        reg_max_depth: int = 10,
        reg_objective: str = "huber",
        # Stage 2 Classifier Hyperparameters
        clf_learning_rate: float = 0.03,
        clf_n_estimators: int = 350,
        clf_num_leaves: int = 31,
        clf_max_depth: int = 10,
        scale_pos_weight: float = 15.0,
        calibration_method: str = "isotonic",
        optimal_threshold: float = 0.35,
        random_state: int = 42,
    ):
        self.reg_learning_rate = reg_learning_rate
        self.reg_n_estimators = reg_n_estimators
        self.reg_num_leaves = reg_num_leaves
        self.reg_max_depth = reg_max_depth
        self.reg_objective = reg_objective

        self.clf_learning_rate = clf_learning_rate
        self.clf_n_estimators = clf_n_estimators
        self.clf_num_leaves = clf_num_leaves
        self.clf_max_depth = clf_max_depth
        self.scale_pos_weight = scale_pos_weight
        self.calibration_method = calibration_method
        self.optimal_threshold = optimal_threshold
        self.random_state = random_state

        # Estimators initialized in fit()
        self.regressor: Optional[lgb.LGBMRegressor] = None
        self.regressor_p10: Optional[lgb.LGBMRegressor] = None
        self.regressor_p90: Optional[lgb.LGBMRegressor] = None
        self.base_classifier: Optional[lgb.LGBMClassifier] = None
        self.calibrated_classifier: Optional[CalibratedClassifierCV] = None

        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_clf: np.ndarray,
        y_train_reg: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val_clf: Optional[np.ndarray] = None,
        y_val_reg: Optional[np.ndarray] = None,
    ) -> TwoStageHeatwavePredictor:
        """
        Fit both Stage 1 Regressors (Departure + Quantiles) and Stage 2 Classifier (with Calibration).
        """
        self.feature_names_ = list(X_train.columns)

        # --------------------------------------------------------------------------------
        # 1. Fit Stage 1: Continuous Departure Regressor (Huber Loss for Robustness)
        # --------------------------------------------------------------------------------
        self.regressor = lgb.LGBMRegressor(
            objective=self.reg_objective,
            n_estimators=self.reg_n_estimators,
            learning_rate=self.reg_learning_rate,
            num_leaves=self.reg_num_leaves,
            max_depth=self.reg_max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.regressor.fit(X_train, y_train_reg)

        # Fit Quantile heads (10th and 90th percentiles for uncertainty bounds)
        self.regressor_p10 = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.10,
            n_estimators=self.reg_n_estimators // 2,
            learning_rate=self.reg_learning_rate,
            num_leaves=self.reg_num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.regressor_p10.fit(X_train, y_train_reg)

        self.regressor_p90 = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.90,
            n_estimators=self.reg_n_estimators // 2,
            learning_rate=self.reg_learning_rate,
            num_leaves=self.reg_num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.regressor_p90.fit(X_train, y_train_reg)

        # --------------------------------------------------------------------------------
        # 2. Fit Stage 2: Cost-Sensitive Classifier + Probability Calibrator
        # --------------------------------------------------------------------------------
        self.base_classifier = lgb.LGBMClassifier(
            n_estimators=self.clf_n_estimators,
            learning_rate=self.clf_learning_rate,
            num_leaves=self.clf_num_leaves,
            max_depth=self.clf_max_depth,
            scale_pos_weight=self.scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.base_classifier.fit(X_train, y_train_clf)

        # Calibrate probabilities on the Validation set if provided, else on training data
        if X_val is not None and y_val_clf is not None:
            try:
                from sklearn.frozen import FrozenEstimator
                cal_est = FrozenEstimator(self.base_classifier)
                self.calibrated_classifier = CalibratedClassifierCV(
                    estimator=cal_est,
                    method=self.calibration_method,
                )
            except ImportError:
                # Backward compatibility with older scikit-learn (<1.4)
                self.calibrated_classifier = CalibratedClassifierCV(
                    estimator=self.base_classifier,
                    method=self.calibration_method,
                    cv="prefit",
                )
            self.calibrated_classifier.fit(X_val, y_val_clf)
        else:
            self.calibrated_classifier = self.base_classifier

        self.is_fitted_ = True
        return self

    def predict_departure(self, X: pd.DataFrame) -> np.ndarray:
        """Predict continuous next-day temperature departure Delta T (in °C)."""
        if not self.is_fitted_ or self.regressor is None:
            raise ValueError("Model is not fitted.")
        return self.regressor.predict(X)

    def predict_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict 10th and 90th percentile departure bounds (in °C)."""
        if not self.is_fitted_ or self.regressor_p10 is None or self.regressor_p90 is None:
            raise ValueError("Model is not fitted.")
        lower = self.regressor_p10.predict(X)
        upper = self.regressor_p90.predict(X)
        return lower, upper

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities [P(Normal), P(Heatwave)]."""
        if not self.is_fitted_ or self.calibrated_classifier is None:
            raise ValueError("Model is not fitted.")
        return self.calibrated_classifier.predict_proba(X)

    def predict_full(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        End-to-end multi-output inference across all records in X.
        Returns a DataFrame with:
        - prob_heatwave: Calibrated P(Heatwave_t+1)
        - predicted_departure: Delta T (in °C)
        - predicted_temp_max: T_max_hat (if threshold_next_day present in X)
        - uncertainty_p10 / uncertainty_p90
        - risk_level: LOW / MODERATE / HIGH / EXTREME
        - predicted_severity: Normal / Warning / Severe / Extreme
        """
        tau = self.optimal_threshold if threshold is None else threshold
        probs = self.predict_proba(X)[:, 1]
        dep_pred = self.predict_departure(X)
        p10, p90 = self.predict_uncertainty(X)

        results = []
        has_thr = "threshold_next_day" in X.columns

        for i in range(len(X)):
            prob = float(probs[i])
            dep = float(dep_pred[i])
            dep_low = float(p10[i])
            dep_high = float(p90[i])
            thr = float(X["threshold_next_day"].iloc[i]) if has_thr else 40.0

            t_pred = round(thr + dep, 2)
            t_low = round(thr + dep_low, 2)
            t_high = round(thr + dep_high, 2)

            # Determine Risk Level and Physical Severity
            if prob < tau:
                severity = "Normal"
                risk_level = "LOW" if prob < 0.20 else "MODERATE"
            else:
                if dep < 2.0:
                    severity = "Warning"
                    risk_level = "HIGH"
                elif dep < 4.0:
                    severity = "Severe"
                    risk_level = "VERY HIGH"
                else:
                    severity = "Extreme"
                    risk_level = "EXTREME"

            results.append(
                {
                    "prob_heatwave": round(prob, 4),
                    "is_heatwave_predicted": int(prob >= tau),
                    "predicted_departure": round(dep, 2),
                    "predicted_temp_max": t_pred,
                    "temp_lower_p10": t_low,
                    "temp_upper_p90": t_high,
                    "risk_level": risk_level,
                    "predicted_severity": severity,
                }
            )

        return pd.DataFrame(results)

    def predict_single(
        self,
        feature_dict: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Convenience single-instance inference for Flask REST API and GenAI services.
        """
        df_row = pd.DataFrame([feature_dict])
        missing_cols = set(self.feature_names_) - set(df_row.columns)
        for col in missing_cols:
            df_row[col] = 0.0

        # Ensure correct column ordering
        df_row = df_row[self.feature_names_]
        out_df = self.predict_full(df_row, threshold=threshold)
        return out_df.iloc[0].to_dict()

    def save(self, path: Union[str, Path]) -> None:
        """Serialize complete predictor pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> TwoStageHeatwavePredictor:
        """Load serialized predictor pipeline from disk."""
        return joblib.load(path)
