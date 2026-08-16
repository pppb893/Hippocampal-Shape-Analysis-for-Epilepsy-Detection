import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, facecolor='#f9d1a3'):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, horizontalalignment='center', verticalalignment='center', fontsize=9)

def draw_arrow(ax, x, y, dx, dy):
    ax.arrow(x, y, dx, dy, head_width=0.4, head_length=0.4, fc='black', ec='black', length_includes_head=True)

def main():
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 25)
    ax.axis('off')

    # Start coordinates
    x_center = 5
    y = 24
    box_w = 4
    box_h = 1
    gap = 1.5

    # Input
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Input: PLS-DA Components (N=10)", facecolor='#d3e0ea')
    draw_arrow(ax, x_center, y, 0, -gap + box_h)
    y -= gap

    # Initial Block
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Conv1D (k=3, c=32) + BN + ReLU", facecolor='#f9d1a3')
    draw_arrow(ax, x_center, y, 0, -gap + box_h)
    y -= gap
    
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "MaxPool1D (k=2, stride=2)", facecolor='#f9d1a3')
    draw_arrow(ax, x_center, y, 0, -gap + box_h)
    y -= gap

    # Residual Block 1
    y_start_b1 = y + box_h
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Conv1D (k=3, c=32) + BN + ReLU", facecolor='#f2a87a')
    draw_arrow(ax, x_center, y, 0, -gap + box_h)
    y -= gap
    
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Conv1D (k=3, c=32) + BN", facecolor='#f2a87a')
    
    # Skip connection 1 (Identity)
    ax.plot([x_center + box_w/2, x_center + box_w/2 + 1], [y_start_b1, y_start_b1], color='black')
    ax.plot([x_center + box_w/2 + 1, x_center + box_w/2 + 1], [y + box_h/2, y_start_b1], color='black')
    draw_arrow(ax, x_center + box_w/2 + 1, y + box_h/2, -1, 0)
    
    # Add node 1
    circle1 = patches.Circle((x_center, y - 0.5), 0.2, facecolor='white', edgecolor='black')
    ax.add_patch(circle1)
    ax.text(x_center, y - 0.5, '+', ha='center', va='center')
    draw_arrow(ax, x_center, y, 0, -0.3)
    draw_arrow(ax, x_center, y - 0.7, 0, -0.6)
    y -= gap

    # Residual Block 2
    y_start_b2 = y + box_h
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Conv1D (k=3, c=64, s=2) + BN + ReLU", facecolor='#ea7c54')
    draw_arrow(ax, x_center, y, 0, -gap + box_h)
    y -= gap
    
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Conv1D (k=3, c=64) + BN", facecolor='#ea7c54')
    
    # Skip connection 2 (Conv)
    ax.plot([x_center + box_w/2, x_center + box_w/2 + 1.5], [y_start_b2, y_start_b2], color='black')
    draw_box(ax, x_center + box_w/2 + 0.5, y + gap/2 + 0.2, 2.5, box_h*0.8, "Conv1D (k=1, c=64, s=2) + BN", facecolor='#ea7c54')
    ax.plot([x_center + box_w/2 + 1.5, x_center + box_w/2 + 1.5], [y + box_h/2, y_start_b2], color='black')
    draw_arrow(ax, x_center + box_w/2 + 1.5, y + box_h/2, -1.5, 0)

    # Add node 2
    circle2 = patches.Circle((x_center, y - 0.5), 0.2, facecolor='white', edgecolor='black')
    ax.add_patch(circle2)
    ax.text(x_center, y - 0.5, '+', ha='center', va='center')
    draw_arrow(ax, x_center, y, 0, -0.3)
    draw_arrow(ax, x_center, y - 0.7, 0, -0.6)
    y -= gap

    # Residual Block 3
    y_start_b3 = y + box_h
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Conv1D (k=3, c=128, s=2) + BN + ReLU", facecolor='#c8523c')
    draw_arrow(ax, x_center, y, 0, -gap + box_h)
    y -= gap
    
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Conv1D (k=3, c=128) + BN", facecolor='#c8523c')
    
    # Skip connection 3 (Conv)
    ax.plot([x_center + box_w/2, x_center + box_w/2 + 1.5], [y_start_b3, y_start_b3], color='black')
    draw_box(ax, x_center + box_w/2 + 0.5, y + gap/2 + 0.2, 2.8, box_h*0.8, "Conv1D (k=1, c=128, s=2) + BN", facecolor='#c8523c')
    ax.plot([x_center + box_w/2 + 1.5, x_center + box_w/2 + 1.5], [y + box_h/2, y_start_b3], color='black')
    draw_arrow(ax, x_center + box_w/2 + 1.5, y + box_h/2, -1.5, 0)

    # Add node 3
    circle3 = patches.Circle((x_center, y - 0.5), 0.2, facecolor='white', edgecolor='black')
    ax.add_patch(circle3)
    ax.text(x_center, y - 0.5, '+', ha='center', va='center')
    draw_arrow(ax, x_center, y, 0, -0.3)
    draw_arrow(ax, x_center, y - 0.7, 0, -0.6)
    y -= gap

    # Output
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Global Average Pooling 1D", facecolor='#a9b5c2')
    draw_arrow(ax, x_center, y, 0, -gap + box_h)
    y -= gap
    
    draw_box(ax, x_center - box_w/2, y, box_w, box_h, "Fully Connected (1 unit) + Sigmoid", facecolor='#5c7a9e')

    plt.title("Custom 1D ResNet-8 Architecture", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('resnet_architecture.png', dpi=300, bbox_inches='tight')
    print("Saved resnet_architecture.png")

if __name__ == '__main__':
    main()
