#  Parameters
batch_size = 10
n_epochs = 100
discriminator_weight = 1.0
l1_weight = 100.0
clip_weight = 10.0

generator_lr = 0.005
discriminator_lr = 0.00005
beta1 = 0.5

# Control
train_model = True

# Paths
images_path = "./Datasets/maps/train"  # cityscapes, facades, maps
test_images_path = "./Datasets/maps/val"  # # cityscapes, facades, maps
results_path = "./Results/maps"  # # cityscapes, facades, maps
