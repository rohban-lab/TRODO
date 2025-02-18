import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms.functional import to_pil_image
import seaborn as sns

def draw_dists(dist1, dist2):
    plt.figure(figsize=(10, 6))

    sns.histplot(dist1, bins=20, color='blue', label='Clean Models Scores', kde=True, stat="density", alpha=0.5)
    sns.histplot(dist2, bins=20, color='orange', label='Trojaned Models scores', kde=True, stat="density", alpha=0.5)

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Scores for Trojaned Models and Clean models')
    plt.legend()

    # Display the plot
    plt.show()


def plot_images_by_label(image_dict):
    plt.clf()
    
    # Determine the number of labels and the maximum number of images in any label
    num_labels = len(image_dict)
    max_images = max(len(images) for images in image_dict.values())
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=max_images, ncols=num_labels, figsize=(num_labels * 2, max_images * 2))
    
    # Flatten the axes array for easy indexing
    if num_labels == 1:
        axes = [[ax] for ax in axes]
    elif max_images == 1:
        axes = [axes]
    
    # Loop through each label and its corresponding images
    for col_index, (label, images) in enumerate(image_dict.items()):
        for row_index, image in enumerate(images):
            ax = axes[row_index][col_index]
            ax.imshow(image)  # Display the image
            ax.axis('off')  # Hide the axes
            if row_index == 0:
                ax.set_title(label)  # Set the title for the first image in each column

    # Adjust layout
    plt.tight_layout()
    plt.show()
    plt.savefig("samples.png")

def visualize_samples(dataloader, n, title="Samples for each label", max_batches=None):

    def to_3_channels(image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        return image

    all_labels = set([y for _, y in dataloader.dataset])
    labels = list(all_labels)    
    
    to_collect_samples = {l:[] for l in range(len(labels))}
    collected_sample = {}
    
    # Collect n x n samples
    for images, targets in dataloader:
        for i, l in zip(images, targets):
            l = l.item()
            image = to_3_channels(i)
            if l in to_collect_samples:
                to_collect_samples[l].append(to_pil_image(image))
                
                if len(to_collect_samples[l]) == n:
                    collected_sample[l] = to_collect_samples[l]
                    to_collect_samples.pop(l, None)
                    
        if len(collected_sample) == len(labels):
            break
    plot_images_by_label(collected_sample)
    


def plot_gaps(cleans, bads, dataset, best_eps, verbose=False):
    
    plt.clf()
    # Plot both arrays
    x = np.arange(len(cleans))  # the label locations

    width = 0.35  
    
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    for i in range(len(cleans)):
        ax.text(x[i], -0.1, f"Eps: {best_eps[i]:.3f}", ha='center')
        
    rects1 = ax.bar(x - width/2, cleans, width, label='clean resnet', color='blue')
    rects2 = ax.bar(x + width/2, bads, width, label='bad resnet', color='red')

    ax.set_ylabel('auroc')
    ax.set_title(f'Bar Chart for {dataset} for best eps')
    ax.set_xticks(x)
    ax.legend()

    fig.savefig(f'results/{dataset}.png', bbox_inches='tight')
    if verbose:
        plt.show()


def plot_process(epsilons, gaps, title, verbose=False):
    plt.clf()
    
    plt.plot(epsilons, gaps)  # Plot y1 with blue color
    plt.xlabel('epsilon')
    plt.ylabel('auroc gap')
    plt.title(title)
    plt.savefig(f'results/{title}.png')
    if verbose:
        plt.show()



def plot_tsne(features, labels):
    
    num_classes = len(set(labels))
    tsne = TSNE(n_components=2).fit_transform(features)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='tab20')  # Adjust the colormap as needed
    plt.colorbar(boundaries=np.arange(12)-0.5).set_ticks(np.arange(11))
    plt.title('TSNE Embedding')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.savefig("my_tsne_plot.png")

def plot_umap(features, labels):
    num_classes = len(set(labels))

    umap_emb = umap.UMAP().fit_transform(features)

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=labels, cmap='tab20')  # Adjust the colormap as needed
    plt.colorbar(boundaries=np.arange(12)-0.5).set_ticks(np.arange(11))
    plt.title('UMAP Embedding')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig("my_umap_plot.png")