import numpy as np
import matplotlib.pyplot as plt

def create_matrix_mult_viz():
    # Create example matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.dot(A, B)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot matrices with custom colors and annotations
    im1 = ax1.imshow(A, cmap='viridis')
    ax1.set_title('Matrix A')
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(A[i, j]), ha='center', va='center', color='white')
    
    im2 = ax2.imshow(B, cmap='viridis')
    ax2.set_title('Matrix B')
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(B[i, j]), ha='center', va='center', color='white')
    
    im3 = ax3.imshow(C, cmap='viridis')
    ax3.set_title('A × B')
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(C[i, j]), ha='center', va='center', color='white')
    
    # Add operation symbols
    fig.text(0.31, 0.5, '×', fontsize=20, ha='center', va='center')
    fig.text(0.64, 0.5, '=', fontsize=20, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('matrix_mult.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_matrix_mult_viz() 