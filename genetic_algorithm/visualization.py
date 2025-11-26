import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from crease2d_ga_functions import genes_to_struc_features

# Default constants for Hollow Tubes
HOLLOW_TUBES_CONFIG = {
    'feature_names': [
        "Tube Diameter",
        "Mean Eccentricity", 
        "Std Eccentricity",
        "Orientation Angle",
        "Kappa Exponent",
        "Cone Angle",
        "Herd Diameter",
        "Herd Length",
        "Herd Extra Nodes"
    ],
    'feature_units': ['Å', '', '', '°', '', '°', '', '', ''],
    'feature_ranges': [
        (100, 400),    # Tube Diameter
        (0, 1),        # Mean Eccentricity
        (0, 1),        # Std Eccentricity
        (0, 90),       # Orientation Angle
        (-5, 5),       # Kappa Exponent
        (0, 90),       # Cone Angle
        (0, 0.5),      # Herd Diameter
        (0, 1),        # Herd Length
        (0, 5)         # Herd Extra Nodes
    ]
}

# Constants for Ellipsoids
ELLIPSOIDS_CONFIG = {
    'feature_names': [
        "Mean Radius",
        "Std Radius", 
        "Mean Aspect Ratio",
        "Std Aspect Ratio",
        "log10 Kappa",
        "Volume Fraction"
    ],
    'feature_units': ['Å', 'Å', '', '', '', ''],
    'feature_ranges': [None, None, None, None, None, None]
}

# Color scheme
COLORS = {
    'violin': '#5a6c7d',
    'edge': '#2c3e50',
    'mean': '#e74c3c',
    'median': '#27ae60',
    'text': '#34495e',
    'bg': '#fafbfc',
    'box_bg': '#f5f7fa',
    'box_edge': '#d1d5db',
    'grid': '#d1d5db'
}

def get_model_config(model_config=None):
    """Get the appropriate feature configuration based on model type."""
    if model_config is not None and hasattr(model_config, 'get_feature_titles'):
        # Use model-specific configuration
        titles = model_config.get_feature_titles()
        if 'radius' in titles[0].lower():  # Ellipsoids
            return ELLIPSOIDS_CONFIG
    # Default to Hollow Tubes
    return HOLLOW_TUBES_CONFIG

def calculate_statistics(data):
    """Calculate statistical measures for the data."""
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }


def format_stats_text(stats, unit):
    """Format statistics text with appropriate precision and units."""
    precision = 2 if unit else 3
    unit_str = f' {unit}' if unit else ''
    
    return (
        f'Mean: {stats["mean"]:.{precision}f}{unit_str}\n'
        f'Median: {stats["median"]:.{precision}f}{unit_str}\n'
        f'Std: {stats["std"]:.{precision}f}{unit_str}\n'
        f'IQR: [{stats["q25"]:.{precision}f}, {stats["q75"]:.{precision}f}]{unit_str}\n'
        f'Range: [{stats["min"]:.{precision}f}, {stats["max"]:.{precision}f}]{unit_str}'
    )


def style_violin_plot(parts):
    """Apply consistent styling to violin plot components."""
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['violin'])
        pc.set_alpha(0.75)
        pc.set_edgecolor(COLORS['edge'])
        pc.set_linewidth(2)
    
    parts['cmeans'].set_color(COLORS['mean'])
    parts['cmeans'].set_linewidth(3)
    parts['cmeans'].set_linestyle('--')
    parts['cmedians'].set_color(COLORS['median'])
    parts['cmedians'].set_linewidth(3)
    
    for key in ['cmaxes', 'cmins']:
        parts[key].set_color(COLORS['edge'])
        parts[key].set_linewidth(2)
    
    parts['cbars'].set_color(COLORS['edge'])
    parts['cbars'].set_linewidth(1.5)


def style_subplot(ax, feature_name, unit, stats, feature_range):
    """Apply consistent styling to subplot axis."""
    # Title
    ax.set_title(feature_name, fontsize=14, fontweight='bold', 
                pad=12, color=COLORS['edge'])
    
    # Y-axis label
    ylabel = f'Value ({unit})' if unit else 'Value'
    ax.set_ylabel(ylabel, fontsize=11, fontweight='600', color=COLORS['text'])
    
    # Remove x-axis ticks
    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)
    
    # Y-axis limits with padding
    if feature_range and all(val is not None for val in feature_range):
        lower, upper = feature_range
    else:
        lower, upper = stats['min'], stats['max']

    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        center = stats['mean'] if np.isfinite(stats['mean']) else 0.0
        spread = abs(center) * 0.1 + 0.1
        lower, upper = center - spread, center + spread
    else:
        span = upper - lower
        padding = span * 0.1 if span > 0 else max(abs(upper), 1.0) * 0.1
        lower -= padding
        upper += padding

    ax.set_ylim(lower, upper)
    
    # Grid and styling
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1, color=COLORS['grid'])
    ax.tick_params(axis='y', labelsize=10, colors=COLORS['text'])
    ax.set_facecolor(COLORS['bg'])
    
    # Borders
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['box_edge'])
        spine.set_linewidth(1.5)


def create_violin_plot_figure(structural_features, numgenes, model_config=None):
    """Create the complete violin plot figure."""
    config = get_model_config(model_config)
    feature_names = config['feature_names']
    feature_units = config['feature_units'] 
    feature_ranges = config['feature_ranges']
    
    # Adjust for actual number of features
    num_features = min(numgenes, structural_features.shape[1], len(feature_names))
    
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                         left=0.08, right=0.95, top=0.94, bottom=0.05)
    
    fig.suptitle('Final Generation Structural Feature Distribution',
                fontsize=20, fontweight='bold', color=COLORS['edge'])
    
    for i in range(num_features):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        feature_data = structural_features[:, i]
        if "log10" in feature_names[i].lower():
            feature_data = np.log10(np.clip(feature_data, 1e-12, None))
        
        # Create violin plot
        parts = ax.violinplot([feature_data], positions=[0],
                             showmeans=True, showextrema=True,
                             showmedians=True, widths=0.8)
        
        # Apply styling
        style_violin_plot(parts)
        
        # Calculate and display statistics
        stats = calculate_statistics(feature_data)
        stats_text = format_stats_text(stats, feature_units[i])
        
        ax.text(0.97, 0.97, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.8',
                        facecolor=COLORS['box_bg'],
                        edgecolor=COLORS['box_edge'],
                        alpha=0.9, linewidth=1.5),
               family='monospace')
        
        # Style subplot
        style_subplot(ax, feature_names[i], feature_units[i], 
                     stats, feature_ranges[i])
    
    return fig


def generate_violin_plots(gatable_final, numgenes, model_config=None):
    """
    Generate violin plots and return as base64 encoded image.
    
    Args:
        gatable_final: GA table with gene values and fitness
        numgenes: Number of genes (features)
        model_config: Model configuration object
    
    Returns:
        Base64 encoded PNG image string
    """
    gene_values_normalized = gatable_final[:, :numgenes]
    
    # Use model-specific conversion if available, otherwise fall back to legacy
    if model_config is not None:
        structural_features = model_config.genes_to_struc_features(gene_values_normalized)
    else:
        structural_features = genes_to_struc_features(gene_values_normalized)
    
    fig = create_violin_plot_figure(structural_features, numgenes, model_config)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return image_base64


def save_violin_plots(gatable_final, numgenes, output_path, model_config=None):
    """
    Generate and save violin plots for the final GA generation.
    
    Args:
        gatable_final: GA table with gene values and fitness
        numgenes: Number of genes (features)
        output_path: Path to save the output image
        model_config: Model configuration object for model-specific gene conversion
    """
    gene_values_normalized = gatable_final[:, :numgenes]
    
    # Use model-specific conversion if available, otherwise fall back to legacy
    if model_config is not None:
        structural_features = model_config.genes_to_struc_features(gene_values_normalized)
    else:
        structural_features = genes_to_struc_features(gene_values_normalized)
    
    fig = create_violin_plot_figure(structural_features, numgenes, model_config)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Violin plots saved to: {output_path}", flush=True)
