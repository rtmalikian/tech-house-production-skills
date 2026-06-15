"""
Sanity Checker — compares current results against previous runs.

Loads historical optimization logs and flags outliers.
"""

import os
import json
import glob
import numpy as np


class SanityChecker:
    """Compares stem metrics against historical runs."""

    def __init__(self, log_dir: str = None):
        self.historical_data = {}
        if log_dir:
            self.load_historical(log_dir)

    def load_historical(self, log_dir: str):
        """Load all optimization_log.json files from a directory."""
        log_files = glob.glob(os.path.join(log_dir, "**", "optimization_log.json"), recursive=True)
        for log_file in log_files:
            try:
                with open(log_file) as f:
                    data = json.load(f)
                for stem in data.get('stems', []):
                    name = stem['stem_name']
                    if name not in self.historical_data:
                        self.historical_data[name] = []
                    self.historical_data[name].append(stem)
            except Exception:
                continue

    def load_pool(self, log_files: list):
        """Load optimization logs from a list of file paths."""
        for log_file in log_files:
            try:
                with open(log_file) as f:
                    data = json.load(f)
                for stem in data.get('stems', []):
                    name = stem['stem_name']
                    if name not in self.historical_data:
                        self.historical_data[name] = []
                    self.historical_data[name].append(stem)
            except Exception:
                continue

    def check_stem(self, stem_name: str, current_metrics: dict) -> dict:
        """
        Check current stem metrics against historical data.

        Args:
            stem_name: stem filename
            current_metrics: dict with lufs, crest_db, peak_db, stereo_corr

        Returns:
            dict with 'status' ('ok' or 'warn'), 'details' list of messages
        """
        details = []
        status = 'ok'

        # Try exact match first, then pattern match
        historical = self.historical_data.get(stem_name, [])
        if not historical:
            # Try matching by role pattern
            role = _detect_role_pattern(stem_name)
            for name, entries in self.historical_data.items():
                if _detect_role_pattern(name) == role:
                    historical = entries
                    break

        if not historical:
            return {'status': 'ok', 'details': ['No historical data for comparison']}

        # Compute historical statistics
        metrics_keys = ['lufs', 'crest_db', 'peak_db', 'stereo_corr']
        for key in metrics_keys:
            values = [s['best_metrics'].get(key) for s in historical
                      if s.get('best_metrics', {}).get(key) is not None]
            if not values or key not in current_metrics:
                continue

            current_val = current_metrics[key]
            median = np.median(values)
            std = np.std(values)
            std = max(std, 0.5)  # minimum std to avoid false positives

            # Check if current value is outside 2 sigma
            if abs(current_val - median) > 2 * std:
                status = 'warn'
                details.append(
                    f"{key}={current_val:+.1f} (historical: {median:+.1f} ± {std:.1f}) [OUTLIER]"
                )
            else:
                details.append(
                    f"{key}={current_val:+.1f} (historical: {median:+.1f} ± {std:.1f}) [OK]"
                )

        return {'status': status, 'details': details}


def _detect_role_pattern(name: str) -> str:
    """Detect role pattern from filename for grouping."""
    name_lower = name.lower()
    if 'kick' in name_lower:
        return 'kick'
    elif 'snare' in name_lower:
        return 'snare'
    elif any(x in name_lower for x in ['hat', 'clap', 'tambourine', 'maracas']):
        return 'percussion'
    elif 'bass' in name_lower:
        return 'bass'
    elif any(x in name_lower for x in ['melody', 'lead']):
        return 'melody'
    elif any(x in name_lower for x in ['counter']):
        return 'counter'
    elif any(x in name_lower for x in ['pad', 'chord']):
        return 'pad'
    elif 'chorus' in name_lower:
        return 'chorus'
    elif 'fx' in name_lower:
        return 'fx'
    return 'other'


def build_pool_report(log_files: list) -> dict:
    """
    Build aggregate statistics from multiple optimization logs.

    Args:
        log_files: list of paths to optimization_log.json files

    Returns:
        dict with per-role statistics
    """
    checker = SanityChecker()
    checker.load_pool(log_files)

    report = {}
    for stem_name, entries in checker.historical_data.items():
        role = _detect_role_pattern(stem_name)
        if role not in report:
            report[role] = {
                'count': 0,
                'metrics': {k: [] for k in ['lufs', 'crest_db', 'peak_db', 'stereo_corr']},
            }

        report[role]['count'] += len(entries)
        for entry in entries:
            metrics = entry.get('best_metrics', {})
            for key in report[role]['metrics']:
                if key in metrics and metrics[key] is not None:
                    report[role]['metrics'][key].append(metrics[key])

    # Compute statistics
    for role in report:
        for key in report[role]['metrics']:
            values = report[role]['metrics'][key]
            if values:
                report[role]['metrics'][key] = {
                    'median': round(float(np.median(values)), 2),
                    'std': round(float(np.std(values)), 2),
                    'min': round(float(np.min(values)), 2),
                    'max': round(float(np.max(values)), 2),
                    'count': len(values),
                }
            else:
                report[role]['metrics'][key] = None

    return report
