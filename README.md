# Neural-Stealth-Drone
code and data samples for paper: designing neural stealth drone

First, please prepare your own dataset, a sampled dataset is provided under path Train_evasion_pattern/design_drone_org_verification_images.

Step 1: Train the DCGAN model on your own dataset.

Step 2: Generate enough synthetic images by DCGAN (flexible amount, recommend not less than the dataset).

Step 3: Use the convolutional layers in the DCGAN discriminator as latent feature extractor to process latent feature maps of both images in the dataset (real images) and synthetic images.

Step 4: Use TDA on both latent feature maps of real images and synthetic images, visualise the TDA space.

Step 5: Filter and leave only weak TDA nodes, and count the number of data samples (real images) for different drone models in weak TDA nodes - this shows the hard-to-learn level.

Step 6: Analyse the curvature metric of each drone model, based on the average curvature of their real images.

Step 7: Design new drone canopy with a maximum of curves, but guarantee the surface is flat.

Step 8: 3-D print the canopy parts, assamble the canopy, and collect images for the post-canopy drone in different environment and lighting conditions.

Step 9: Train a dedicated ResNet18 drone classifior on you dataset.

Step 10: Load pre-trained Yolo generic object detectors.

Step 11: Train the evasion pattern, use Yolo and ResNet models on images for post-canopy drone (several pattern model are provided in the code, e.g. train with perspective, train with affine, train without perspective).

Step 12: Print the evasion pattern on a physical patch, stick it on the post-canopy drone, and collect more images for method validation.

Step 13: Validation - apply Yolo and ResNet on new collected images for post-canopy drone to verify the performance of neural stealth drone. Use real-time video could achieve a more accurate evaluation but code not provided here.
