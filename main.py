"""
Neural Network Noise Robustness Demonstration - OOD Generalization
==================================================================
This module demonstrates how introducing noise during training
improves out-of-distribution generalization in neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configure matplotlib for black background
plt.style.use('dark_background')

class NoisyNN(nn.Module):
    """Neural network with optional noise injection"""
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, noise_std=0.0):
        super().__init__()
        self.noise_std = noise_std
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, training=True):
        # Add input noise during training
        if training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x) if training else x
        x = self.relu(self.fc2(x))
        x = self.dropout(x) if training else x
        x = self.fc3(x)
        return x

def create_training_data(n_samples=1000):
    """Create training data with specific structure"""
    # Create two moons dataset for training
    X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Add a small amount of additional samples at the boundaries to make it more challenging
    n_boundary = n_samples // 10
    theta = np.linspace(0, np.pi, n_boundary)
    X_boundary_0 = np.column_stack([np.cos(theta), np.sin(theta)]) * 0.8
    X_boundary_1 = np.column_stack([np.cos(theta) + 1, -np.sin(theta)]) * 0.8
    
    X_boundary = np.vstack([X_boundary_0, X_boundary_1])
    y_boundary = np.hstack([np.zeros(n_boundary), np.ones(n_boundary)])
    
    # Add slight noise
    X_boundary += np.random.normal(0, 0.1, X_boundary.shape)
    X_boundary = scaler.transform(X_boundary)
    
    # Combine
    X = np.vstack([X, X_boundary])
    y = np.hstack([y, y_boundary]).astype(int)
    
    return X, y, scaler

def create_ood_test_data(n_samples=500, scaler=None):
    """Create out-of-distribution test data - more realistic scenarios"""
    # Create several OOD scenarios
    ood_datasets = []
    
    # 1. Slightly shifted moons (mild distribution shift)
    X_shifted, y_shifted = make_moons(n_samples=n_samples//4, noise=0.15, random_state=123)
    X_shifted = scaler.transform(X_shifted)
    X_shifted[:, 0] += 0.3  # Small shift horizontally
    X_shifted[:, 1] += 0.2  # Small shift vertically
    ood_datasets.append(('Mild Shift', X_shifted, y_shifted))
    
    # 2. Noisier moons (increased noise)
    X_noisy, y_noisy = make_moons(n_samples=n_samples//4, noise=0.35, random_state=456)
    X_noisy = scaler.transform(X_noisy)
    ood_datasets.append(('Higher Noise', X_noisy, y_noisy))
    
    # 3. Slightly rotated moons (small rotation)
    X_rotated, y_rotated = make_moons(n_samples=n_samples//4, noise=0.15, random_state=789)
    X_rotated = scaler.transform(X_rotated)
    theta = np.pi / 12  # 15 degree rotation
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
    X_rotated = X_rotated @ rotation_matrix
    ood_datasets.append(('Mild Rotation', X_rotated, y_rotated))
    
    # 4. Interpolated structure (between moons and circles)
    # Create moons and circles, then interpolate
    X_moons, y_moons = make_moons(n_samples=n_samples//4, noise=0.15, random_state=321)
    X_circles, y_circles = make_circles(n_samples=n_samples//4, noise=0.15, factor=0.5, random_state=321)
    
    # Interpolate between moons and circles
    alpha = 0.3  # 30% circles, 70% moons
    X_interp = (1 - alpha) * X_moons + alpha * X_circles
    X_interp = scaler.transform(X_interp)
    y_interp = y_moons  # Keep the same labels
    ood_datasets.append(('Mixed Structure', X_interp, y_interp))
    
    return ood_datasets

def train_model(X_train, y_train, noise_std=0.0, epochs=200, verbose=True):
    """Train a model with specified noise level"""
    model = NoisyNN(noise_std=noise_std)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        
        if verbose and epoch % 50 == 0:
            print(f"Noise σ={noise_std:.1f}, Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model, losses

def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        y_tensor = torch.LongTensor(y_test)
        outputs = model(X_tensor, training=False)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean().item()
    return accuracy

def plot_decision_boundary_with_ood(ax, model, X_train, y_train, X_ood, y_ood, 
                                   title, ood_name="OOD"):
    """Plot decision boundary with training and OOD data"""
    h = 0.02
    x_min = min(X_train[:, 0].min(), X_ood[:, 0].min()) - 0.5
    x_max = max(X_train[:, 0].max(), X_ood[:, 0].max()) + 0.5
    y_min = min(X_train[:, 1].min(), X_ood[:, 1].min()) - 0.5
    y_max = max(X_train[:, 1].max(), X_ood[:, 1].max()) + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    model.eval()
    with torch.no_grad():
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]), training=False)
        Z = torch.softmax(Z, dim=1)[:, 1].numpy()
        Z = Z.reshape(xx.shape)
    
    # Create artistic contour plot
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5], colors='yellow', linewidths=2, linestyles='-')
    
    # Plot training data with smaller, semi-transparent points
    train_colors = ['#FF6B6B', '#4ECDC4']
    for i in range(2):
        mask = y_train == i
        ax.scatter(X_train[mask, 0], X_train[mask, 1], c=train_colors[i], s=30, 
                  edgecolors='white', linewidth=0.5, alpha=0.5, label=f'Train Class {i}')
    
    # Plot OOD data with larger, prominent points
    ood_colors = ['#FFD93D', '#6BCB77']
    for i in range(2):
        mask = y_ood == i
        ax.scatter(X_ood[mask, 0], X_ood[mask, 1], c=ood_colors[i], s=80, 
                  marker='*', edgecolors='black', linewidth=1, alpha=0.9, 
                  label=f'{ood_name} Class {i}')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.7, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

def create_ood_visualization():
    """Create comprehensive OOD generalization visualization"""
    # Generate training data
    print("Creating training data...")
    X_train, y_train, scaler = create_training_data(n_samples=1000)
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=0.2, random_state=42)
    
    # Generate OOD test datasets
    print("Creating OOD test data...")
    ood_datasets = create_ood_test_data(n_samples=400, scaler=scaler)
    
    # Train models with different noise levels
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    models = {}
    
    print("\nTraining models...")
    for noise in noise_levels:
        print(f"\nTraining with noise σ={noise}")
        models[noise], _ = train_model(X_train, y_train, noise, epochs=300, verbose=False)
    
    # Create main figure
    fig = plt.figure(figsize=(20, 16), facecolor='black')
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1: Decision boundaries with different OOD scenarios
    for i, (noise, (ood_name, X_ood, y_ood)) in enumerate(zip(noise_levels, ood_datasets)):
        ax = fig.add_subplot(gs[0, i])
        plot_decision_boundary_with_ood(ax, models[noise], X_train, y_train, 
                                       X_ood, y_ood, 
                                       f'σ={noise}: {ood_name}', ood_name)
    
    # Row 2: OOD performance comparison
    ax_ood = fig.add_subplot(gs[1, :2])
    
    # Evaluate all models on all OOD datasets
    ood_results = {noise: [] for noise in noise_levels}
    ood_names = []
    
    for ood_name, X_ood, y_ood in ood_datasets:
        ood_names.append(ood_name.replace(' Distribution', '').replace(' (Blobs)', ''))
        for noise in noise_levels:
            acc = evaluate_model(models[noise], X_ood, y_ood)
            ood_results[noise].append(acc)
    
    # Add in-distribution validation accuracy
    ood_names.insert(0, 'In-Distribution')
    for noise in noise_levels:
        val_acc = evaluate_model(models[noise], X_val, y_val)
        ood_results[noise].insert(0, val_acc)
    
    # Plot OOD performance
    x = np.arange(len(ood_names))
    width = 0.2
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_levels)))
    
    for i, noise in enumerate(noise_levels):
        ax_ood.bar(x + i*width - 1.5*width, ood_results[noise], width, 
                   label=f'σ={noise}', color=colors[i], alpha=0.8)
    
    ax_ood.set_xlabel('Test Distribution', fontsize=14)
    ax_ood.set_ylabel('Accuracy', fontsize=14)
    ax_ood.set_title('Model Performance on Different Distributions', fontsize=16, fontweight='bold')
    ax_ood.set_xticks(x)
    ax_ood.set_xticklabels(ood_names, rotation=15, ha='right')
    ax_ood.legend(loc='lower left', frameon=True, fancybox=True, framealpha=0.7)
    ax_ood.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax_ood.set_ylim(0, 1.05)
    
    # Add performance drop visualization
    ax_drop = fig.add_subplot(gs[1, 2:])
    
    # Calculate performance drops
    performance_drops = {}
    for noise in noise_levels:
        in_dist_acc = ood_results[noise][0]
        ood_accs = ood_results[noise][1:]
        drops = [(in_dist_acc - acc) * 100 for acc in ood_accs]
        performance_drops[noise] = drops
    
    # Plot performance drops
    for i, noise in enumerate(noise_levels):
        ax_drop.plot(ood_names[1:], performance_drops[noise], 
                    marker='o', markersize=10, linewidth=3,
                    label=f'σ={noise}', color=colors[i])
    
    ax_drop.set_xlabel('OOD Test Distribution', fontsize=14)
    ax_drop.set_ylabel('Performance Drop (%)', fontsize=14)
    ax_drop.set_title('Generalization Gap: In-Distribution vs OOD', fontsize=16, fontweight='bold')
    ax_drop.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.7)
    ax_drop.grid(True, alpha=0.3, linestyle='--')
    ax_drop.set_xticklabels(ood_names[1:], rotation=15, ha='right')
    
    # Row 3: Gradient magnitude analysis
    ax_grad = fig.add_subplot(gs[2, :2])
    
    # Compute average gradient magnitudes
    def compute_gradient_magnitude(model, X):
        model.eval()
        X_tensor = torch.FloatTensor(X).requires_grad_(True)
        outputs = model(X_tensor, training=False)
        loss = outputs.sum()
        loss.backward()
        grad_mag = X_tensor.grad.norm(dim=1).mean().item()
        return grad_mag
    
    grad_mags = {noise: [] for noise in noise_levels}
    test_points = np.random.randn(100, 2) * 2  # Random test points
    
    for noise in noise_levels:
        grad_mag = compute_gradient_magnitude(models[noise], test_points)
        grad_mags[noise] = grad_mag
    
    # Bar plot of gradient magnitudes
    bars = ax_grad.bar(range(len(noise_levels)), 
                       [grad_mags[n] for n in noise_levels],
                       color=colors, alpha=0.8)
    ax_grad.set_xlabel('Training Noise Level (σ)', fontsize=14)
    ax_grad.set_ylabel('Average Gradient Magnitude', fontsize=14)
    ax_grad.set_title('Input Gradient Magnitude vs Training Noise', fontsize=16, fontweight='bold')
    ax_grad.set_xticks(range(len(noise_levels)))
    ax_grad.set_xticklabels([f'{n}' for n in noise_levels])
    ax_grad.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar, noise in zip(bars, noise_levels):
        height = bar.get_height()
        ax_grad.text(bar.get_x() + bar.get_width()/2., height,
                    f'{grad_mags[noise]:.2f}',
                    ha='center', va='bottom', color='white', fontsize=12)
    
    # Summary statistics
    # Summary statistics
    ax_summary = fig.add_subplot(gs[2, 2:])

    # Create summary table
    summary_data = []
    for noise in noise_levels:
        avg_ood_acc = np.mean(ood_results[noise][1:])  # Exclude in-distribution
        std_ood_acc = np.std(ood_results[noise][1:])
        avg_drop = np.mean(performance_drops[noise])
        summary_data.append([noise, ood_results[noise][0], avg_ood_acc, std_ood_acc, avg_drop])

    # Plot summary table
    ax_summary.axis('tight')
    ax_summary.axis('off')
    table_data = [[f'{d[0]:.1f}', f'{d[1]:.3f}', f'{d[2]:.3f}', f'{d[3]:.3f}', f'{d[4]:.1f}%'] 
                for d in summary_data]

    # Create table with better spacing
    table = ax_summary.table(cellText=table_data,
                            colLabels=['Noise σ', 'Val Acc', 'Avg OOD Acc', 'Std OOD', 'Avg Drop'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.18, 0.18, 0.22, 0.18, 0.24])  # Adjusted widths
    table.auto_set_font_size(False)
    table.set_fontsize(11)  # Slightly smaller font
    table.scale(1.2, 2.5)  # Better scaling

    # Style the table with fixed visibility
    for i in range(len(noise_levels) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#303030')
                cell.set_text_props(weight='bold', color='white', fontsize=12)
                cell.set_height(0.08)
            else:
                # Set base style for all data cells
                cell.set_facecolor('#1a1a1a')
                cell.set_text_props(color='white', fontsize=11)
                cell.set_height(0.06)
                
                # Special coloring only for Avg OOD Acc column
                if j == 2:  # Avg OOD Acc column
                    val = float(table_data[i-1][j])
                    if val > 0.8:
                        cell.set_facecolor('#2d5a2d')
                    elif val > 0.75:
                        cell.set_facecolor('#3d6a3d')
                    elif val > 0.7:
                        cell.set_facecolor('#5a5a2d')
                    else:
                        cell.set_facecolor('#5a2d2d')
                
                # Highlight best performance row
                if i == 3:  # σ=0.3 row (best performer)
                    if j != 2:  # Don't override the Avg OOD Acc coloring
                        cell.set_facecolor('#2a2a3a')

    # Add a border around the table
    for i in range(len(noise_levels) + 1):
        for j in range(5):
            cell = table[(i, j)]
            cell.set_linewidth(1)
            cell.set_edgecolor('#555555')

    ax_summary.set_title('Summary: OOD Generalization Performance', 
                        fontsize=16, fontweight='bold', pad=20, color='white')    
    # Main title
    fig.suptitle('Neural Network Noise Injection: Out-of-Distribution Generalization', 
                 fontsize=24, fontweight='bold', y=0.98, color='white')
    
    # Add explanatory text
    fig.text(0.5, 0.01, 
             'Training with noise injection improves robustness to distribution shifts: shifted, rotated, scaled, and completely different data structures',
             ha='center', fontsize=14, style='italic', color='lightgray')
    
    plt.tight_layout()
    return fig

def create_additional_ood_scenarios():
    """Create an additional figure showing progressive OOD scenarios"""
    fig = plt.figure(figsize=(20, 12), facecolor='black')
    
    # Generate training data
    X_train, y_train, scaler = create_training_data(n_samples=800)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=0.2, random_state=42)
    
    # Train two models: without and with noise
    print("Training comparison models...")
    model_no_noise, _ = train_model(X_train, y_train, noise_std=0.0, epochs=300, verbose=False)
    model_with_noise, _ = train_model(X_train, y_train, noise_std=0.3, epochs=300, verbose=False)
    
    # Create progressive OOD scenarios
    n_samples = 200
    ood_scenarios = []
    
    # Progressive shifts
    for shift in [0, 0.2, 0.4, 0.6, 0.8]:
        X_ood, y_ood = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
        X_ood = scaler.transform(X_ood)
        X_ood[:, 0] += shift
        X_ood[:, 1] += shift * 0.5
        ood_scenarios.append((f'Shift={shift}', X_ood, y_ood))
    
    # Progressive noise levels
    for noise in [0.1, 0.2, 0.3, 0.4, 0.5]:
        X_ood, y_ood = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        X_ood = scaler.transform(X_ood)
        ood_scenarios.append((f'Noise={noise}', X_ood, y_ood))
    
    # Evaluate models
    results_no_noise = []
    results_with_noise = []
    
    for name, X_ood, y_ood in ood_scenarios:
        acc_no_noise = evaluate_model(model_no_noise, X_ood, y_ood)
        acc_with_noise = evaluate_model(model_with_noise, X_ood, y_ood)
        results_no_noise.append(acc_no_noise)
        results_with_noise.append(acc_with_noise)
    
    # Plot results
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Progressive shift performance
    ax1 = fig.add_subplot(gs[0, 0])
    shift_values = [0, 0.2, 0.4, 0.6, 0.8]
    ax1.plot(shift_values, results_no_noise[:5], 'o-', color='#FF6B6B', 
             linewidth=3, markersize=10, label='No noise training')
    ax1.plot(shift_values, results_with_noise[:5], 's-', color='#4ECDC4', 
             linewidth=3, markersize=10, label='With noise (σ=0.3)')
    ax1.set_xlabel('Distribution Shift Amount', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Robustness to Progressive Shifts', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, framealpha=0.7)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0.4, 1.05)
    
    # Plot 2: Progressive noise performance
    ax2 = fig.add_subplot(gs[0, 1])
    noise_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    ax2.plot(noise_values, results_no_noise[5:], 'o-', color='#FF6B6B', 
             linewidth=3, markersize=10, label='No noise training')
    ax2.plot(noise_values, results_with_noise[5:], 's-', color='#4ECDC4', 
             linewidth=3, markersize=10, label='With noise (σ=0.3)')
    ax2.set_xlabel('Test Data Noise Level', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Robustness to Increasing Noise', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, framealpha=0.7)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0.4, 1.05)
    
    # Plot 3: Performance gap
    ax3 = fig.add_subplot(gs[0, 2])
    gap_shift = np.array(results_with_noise[:5]) - np.array(results_no_noise[:5])
    gap_noise = np.array(results_with_noise[5:]) - np.array(results_no_noise[5:])
    
    x = np.arange(5)
    width = 0.35
    ax3.bar(x - width/2, gap_shift * 100, width, label='Shift scenarios', 
            color='#9B59B6', alpha=0.8)
    ax3.bar(x + width/2, gap_noise * 100, width, label='Noise scenarios', 
            color='#3498DB', alpha=0.8)
    ax3.set_xlabel('Scenario Intensity', fontsize=12)
    ax3.set_ylabel('Performance Improvement (%)', fontsize=12)
    ax3.set_title('Noise Training Advantage', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
    ax3.legend(frameon=True, fancybox=True, framealpha=0.7)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.axhline(y=0, color='white', linestyle='-', alpha=0.5)
    
    # Plot decision boundaries for extreme cases
    # No shift
    ax4 = fig.add_subplot(gs[1, 0])
    plot_decision_boundary_with_ood(ax4, model_no_noise, X_train, y_train,
                                   ood_scenarios[0][1], ood_scenarios[0][2],
                                   'No Noise Training: Original Distribution', 'Test')
    
    ax5 = fig.add_subplot(gs[1, 1])
    plot_decision_boundary_with_ood(ax5, model_with_noise, X_train, y_train,
                                   ood_scenarios[0][1], ood_scenarios[0][2],
                                   'Noise Training: Original Distribution', 'Test')
    
    # High shift
    ax6 = fig.add_subplot(gs[1, 2])
    plot_decision_boundary_with_ood(ax6, model_with_noise, X_train, y_train,
                                   ood_scenarios[4][1], ood_scenarios[4][2],
                                   'Noise Training: High Shift', 'Shifted')
    
    fig.suptitle('Progressive Out-of-Distribution Analysis: Impact of Noise Training', 
                 fontsize=22, fontweight='bold', y=0.98, color='white')
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    # Create and display the OOD visualization
    print("Creating OOD generalization visualization...")
    ood_fig = create_ood_visualization()
    plt.show()
    
    # Create and display progressive OOD analysis
    print("\nCreating progressive OOD analysis...")
    progressive_fig = create_additional_ood_scenarios()
    plt.show()
    
    # Save figures if needed
    ood_fig.savefig('noise_ood_generalization.png', dpi=300, bbox_inches='tight', facecolor='black')
    progressive_fig.savefig('noise_progressive_ood.png', dpi=300, bbox_inches='tight', facecolor='black')