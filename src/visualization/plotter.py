def plot_mangrove_mask(mask, title='Mangrove Mask'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.imshow(mask, cmap='Greens')
    plt.title(title)
    plt.axis('off')
    plt.colorbar(label='Mask Value')
    plt.show()

def plot_carbon_estimates(estimates, title='Carbon Estimates'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(estimates)), estimates, color='blue')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Estimated Carbon (tonnes)')
    plt.xticks(range(len(estimates)))
    plt.show()