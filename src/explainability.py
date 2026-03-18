"""
SHAP explainability for DrugLens.

Uses SHAP (SHapley Additive exPlanations) to explain individual predictions
by decomposing them into per-feature contributions.
"""

import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def get_shap_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def explain_prediction(
    explainer: shap.TreeExplainer,
    features: np.ndarray,
    feature_names: list[str],
    top_k: int = 15,
) -> dict:
    X = features.reshape(1, -1)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base = explainer.expected_value[1]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
            base = explainer.expected_value[1]
        elif shap_values.ndim == 2:
            sv = shap_values[0]
            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = float(base[0]) if len(base) > 0 else float(base)
        elif shap_values.ndim == 1:
            sv = shap_values
            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = float(base[0]) if len(base) > 0 else float(base)
        else:
            sv = shap_values.flatten()
            base = 0.0
    else:
        sv = np.array(shap_values).flatten()
        base = 0.0

    base = float(base)
    feat_values = features.flatten()

    importance = list(zip(feature_names, sv, feat_values))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    top_positive = [
        {"feature": name, "shap_value": round(float(val), 4), "feature_value": round(float(fv), 4)}
        for name, val, fv in importance if val > 0
    ][:top_k]

    top_negative = [
        {"feature": name, "shap_value": round(float(val), 4), "feature_value": round(float(fv), 4)}
        for name, val, fv in importance if val < 0
    ][:top_k]

    return {
        "shap_values": sv,
        "base_value": base,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 15,
    dark_mode: bool = False,
) -> plt.Figure:
    importance = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_k]

    names = [x[0] for x in importance][::-1]
    values = [x[1] for x in importance][::-1]

    if dark_mode:
        bg = '#1a1a1a'
        text = '#e8e0d8'
        muted = '#8a8078'
        pos_color = '#5a9e6f'
        neg_color = '#c4705a'
    else:
        bg = '#faf8f5'
        text = '#2d2a26'
        muted = '#8a8078'
        pos_color = '#5a9e6f'
        neg_color = '#c4705a'

    colors = [neg_color if v < 0 else pos_color for v in values]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    bars = ax.barh(names, values, color=colors, height=0.6, edgecolor='none')

    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=9, color=muted,
                   fontfamily='sans-serif', labelpad=12)
    ax.axvline(x=0, color=muted, linewidth=0.5, linestyle='-', alpha=0.4)

    ax.tick_params(axis='y', labelsize=8.5, colors=text, length=0, pad=8)
    ax.tick_params(axis='x', labelsize=8, colors=muted, length=3, pad=6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.set_tick_params(width=0.5)
    ax.grid(axis='x', alpha=0.08, color=muted)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=pos_color, label='Favors binding'),
        Patch(facecolor=neg_color, label='Opposes binding'),
    ]
    leg = ax.legend(handles=legend_elements, loc='lower right', frameon=False,
                    fontsize=8, labelcolor=muted)

    plt.tight_layout(pad=1.5)
    return fig
