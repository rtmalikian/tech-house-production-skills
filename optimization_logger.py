"""
Optimization Logger — structured logging for all optimization data.

Logs per-stem optimization results and session summaries to JSON.
"""

import os
import json
from datetime import datetime


class OptimizationLogger:
    """Logs optimization history to JSON files."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "optimization_log.json")
        self.session_data = {
            'timestamp': datetime.now().isoformat(),
            'stems': [],
            'summary': {},
        }

    def log_stem_optimization(self, stem_name: str, role: str,
                              best_params: dict, history: list,
                              sanity_result: dict = None):
        """
        Log optimization results for a single stem.

        Args:
            stem_name: stem filename
            role: stem role (bass, melody, etc.)
            best_params: dict of optimized parameters
            history: list of evaluation dicts from optimizer
            sanity_result: dict from sanity checker (optional)
        """
        # Find best evaluation
        best_eval = min(history, key=lambda h: h.get('loss', 100)) if history else {}

        stem_data = {
            'stem_name': stem_name,
            'role': role,
            'timestamp': datetime.now().isoformat(),
            'best_params': best_params,
            'best_metrics': best_eval.get('metrics', {}),
            'best_loss': best_eval.get('loss', 0.0),
            'total_evaluations': len(history),
            'pass_flags': best_eval.get('pass_flags', {}),
            'all_pass': best_eval.get('all_pass', False),
            'sanity': sanity_result,
            'history': history,
        }

        self.session_data['stems'].append(stem_data)

    def log_session_summary(self):
        """Compute and log session summary."""
        stems = self.session_data['stems']
        if not stems:
            return

        total_evals = sum(s['total_evaluations'] for s in stems)
        avg_evals = total_evals / len(stems) if stems else 0

        # Stems needing most correction (highest loss)
        by_loss = sorted(stems, key=lambda s: s['best_loss'], reverse=True)
        worst_stems = [s['stem_name'] for s in by_loss[:5]]

        # Final spectral balance (average across all stems)
        all_lufs = [s['best_metrics'].get('lufs', 0) for s in stems]
        all_crest = [s['best_metrics'].get('crest_db', 0) for s in stems]

        self.session_data['summary'] = {
            'total_stems': len(stems),
            'total_evaluations': total_evals,
            'avg_evaluations_per_stem': round(avg_evals, 1),
            'stems_needing_most_correction': worst_stems,
            'avg_lufs': round(sum(all_lufs) / len(all_lufs), 1) if all_lufs else 0,
            'avg_crest_db': round(sum(all_crest) / len(all_crest), 1) if all_crest else 0,
            'stems_all_pass': sum(1 for s in stems if s['all_pass']),
            'stems_with_warnings': sum(1 for s in stems if not s['all_pass']),
        }

        self._write_log()

    def _write_log(self):
        """Write session data to JSON file."""
        with open(self.log_path, 'w') as f:
            json.dump(self.session_data, f, indent=2, default=str)

    def get_session_summary(self) -> dict:
        """Get the session summary."""
        return self.session_data.get('summary', {})
