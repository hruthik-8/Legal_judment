import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams['figure.facecolor'] = 'white'

def create_civil_metrics_plot():
    # Sample data (replace with actual metrics if available)
    metrics = {
        'Accuracy': 0.82,
        'F1-Score': 0.80,
        'ROC-AUC': 0.87,
        'Training Loss': 0.42
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot for metrics
    sns.barplot(x=list(metrics.keys())[:3], y=list(metrics.values())[:3], ax=ax1)
    ax1.set_title('Civil Model Classification Metrics', fontsize=12, pad=15)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(list(metrics.values())[:3]):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # Loss plot
    epochs = list(range(1, 6))
    train_loss = [0.65, 0.48, 0.42, 0.41, 0.40]
    val_loss = [0.72, 0.55, 0.50, 0.49, 0.51]
    
    ax2.plot(epochs, train_loss, 'o-', label='Training Loss')
    ax2.plot(epochs, val_loss, 's-', label='Validation Loss')
    ax2.set_title('Training & Validation Loss', fontsize=12, pad=15)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('civil_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_criminal_metrics_plot():
    # Sample data (replace with actual metrics if available)
    metrics = {
        'Charge Accuracy': 0.75,
        'Articles F1': 0.70,
        'Penalty MAE': 4.5,
        'Penalty MSE': 100.0
    }
    
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[2, 1])
    
    # Classification metrics
    ax1 = fig.add_subplot(gs[0])
    sns.barplot(x=list(metrics.keys())[:2], y=list(metrics.values())[:2], ax=ax1)
    ax1.set_title('Criminal Model Classification Metrics', fontsize=12, pad=15)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Score')
    
    # Add value labels
    for i, v in enumerate(list(metrics.values())[:2]):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # Penalty metrics
    ax2 = fig.add_subplot(gs[1])
    penalty_metrics = {
        'MAE (months)': metrics['Penalty MAE'],
        'MSE (months²)': metrics['Penalty MSE']
    }
    
    bars = ax2.bar(penalty_metrics.keys(), penalty_metrics.values())
    ax2.set_title('Penalty Prediction', fontsize=12, pad=15)
    ax2.set_ylabel('Value')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels with different y-scales
    for bar, (k, v) in zip(bars, penalty_metrics.items()):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5, 
                f"{v:.1f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('criminal_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating visualizations...")
    create_civil_metrics_plot()
    create_criminal_metrics_plot()
    print("Visualizations saved as 'civil_metrics.png' and 'criminal_metrics.png'")
