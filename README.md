README
Project Overview
This project focuses on the detection of brain tumors using multimodal imaging data, specifically paired CT and MRI scans. The primary goal is to develop and evaluate a deep learning model capable of accurately classifying whether a patient's scans indicate the presence of a brain tumor or are healthy. The dataset utilized in this project is sourced from Kaggle and comprises paired CT and MRI images of individuals diagnosed with brain tumors and healthy control subjects.

Model Architecture: LightweightDualCrossModalTransformer
The core of this project is the LightweightDualCrossModalTransformer model, a custom deep learning architecture designed for multimodal image classification. The model processes paired CT and MRI scans using parallel pathways that interact through attention mechanisms. Its key components are:

CNN Encoders: Separate Convolutional Neural Network (CNN) encoders are used for each modality (CT and MRI). These encoders process the input images to extract spatial features and transform them into a sequence of patch embeddings. Each encoder consists of multiple convolutional layers with ReLU activation and pooling, progressively reducing spatial dimensions and increasing channel depth. The output of these encoders is a sequence of feature vectors representing different regions (patches) of the input image.

Self-Attention: Following the CNN encoders, each modality's feature sequence is processed by a self-attention mechanism (implemented using nn.TransformerEncoderLayer). Self-attention allows the model to weigh the importance of different patches within the same modality, capturing long-range dependencies and contextual information within the CT and MRI scans independently.

Dual Cross-Attention: This is a critical component for multimodal fusion. The model employs two cross-attention blocks:

CT attending to MRI: Features from the MRI modality act as the Key and Value, while CT features act as the Query. This allows the model to enhance the CT features by incorporating relevant information from the MRI scan.
MRI attending to CT: Conversely, features from the CT modality act as the Key and Value, while MRI features act as the Query. This allows the model to enhance the MRI features by incorporating relevant information from the CT scan. This dual interaction enables a rich exchange of information between the two modalities, creating fused features that leverage the complementary strengths of CT and MRI.
Classification Layer: After the dual cross-attention, the attended feature sequences from both modalities are typically pooled (e.g., Global Average Pooling) to obtain fixed-size representations. These pooled, fused features are then concatenated and passed through a final fully connected layer (or a small stack of linear layers) to output the classification logits, indicating the probability of the input image pair belonging to each class (Healthy or Tumor).

Code Structure
This notebook is structured into several logical sections, covering data loading, preprocessing, model definition, training, evaluation, and visualization of results and model interpretability.

Code Cells and Their Purpose:
Cell VmGPj_RfpCCU: Installs the kaggle library to enable downloading datasets from Kaggle.
Cell dVTl41jcpGK6: Mounts Google Drive to access files, specifically the kaggle.json file needed for Kaggle API authentication.
Cell eq9YC0dhpI89: Configures the Kaggle API by creating the necessary directory, copying the kaggle.json file from Google Drive, and setting the correct file permissions.
Cell IqZv9q2IpMQ2: Downloads the specified Kaggle dataset containing brain tumor CT and MRI multimodal images.
Cell URRFrjKxpPF3: Unzips the downloaded dataset file to make the image data accessible within the Colab environment.
Cell GrqMtJ0brT62: Imports necessary Python libraries for data handling (os, cv2, numpy, matplotlib, pandas), model building (tensorflow), and data splitting/preprocessing (sklearn).
Cell 2tXFJOlTpTMI: Defines the base paths for the CT and MRI image directories and the category names ('Healthy', 'Tumor').
Cell 26d76464: Defines sample image names and constructs their full paths for demonstration or specific access.
Cell MveYRisr0bqE: Defines the load_image_paths_and_labels function and calls it to load image paths and labels for both CT and MRI scans into pandas DataFrames (ct_df, mri_df). Displays the head of the resulting DataFrames.
Cell OU_vZXFsMpH-: Defines the encode_labels function and uses it to encode the categorical labels ('Healthy', 'Tumor') into numerical format (0, 1) for both CT and MRI DataFrames. Displays the head of the DataFrames after encoding.
Cell 2f6827d2: Defines the create_paired_dataframe function and calls it to create a DataFrame (paired_df) containing paired CT and MRI image paths based on filename patterns. Displays the head and shape of the paired DataFrame.
Cell 7cc01600: Defines the custom PyTorch PairedBrainTumorDataset class, which is used to load and preprocess paired images from the DataFrame. It includes logic for reading images, applying transformations, and returning paired image tensors and labels.
Cell 437b9f5c: Defines image transformations (including data augmentation) using torchvision.transforms. Splits the paired_df into training, validation, and testing sets (train_paired_df, valid_paired_df, test_paired_df) using train_test_split with stratification. Creates instances of the PairedBrainTumorDataset for each split.
Cell 54b2f0fc: Defines the visualize_preprocessing function to display sample image pairs before and after applying the defined preprocessing and augmentation transformations. Calls this function to visualize samples from the training dataset.
Cell db1c2fe5: Creates PyTorch DataLoader instances for the training, validation, and testing datasets (train_paired_loader, valid_paired_loader, test_paired_loader) using the previously created datasets.
Cell Q34tveY_se4L: Defines the LightweightDualCrossModalTransformer model architecture using PyTorch modules (torch.nn). This includes the CNNEncoder and CrossAttentionBlock components, and the main model combining them.
Cell 54d50eb2: Defines the train_model_with_history function to handle the training loop (forward pass, loss calculation, backward pass, optimization) and evaluation on the validation set, including early stopping. It also defines plot_training_history to visualize training progress. Instantiates and trains the LightweightDualCrossModalTransformer model using these functions and saves the best model state.
Cell 5480b692: Loads the saved state dictionary into a new instance of the LightweightDualCrossModalTransformer model.
Cell f7c10f21: Defines the select_sample_images_for_visualization function to select sample image pairs from the test loader and collect their paths, true labels, predicted labels, and attention weights for visualization. Calls this function to populate sample_results.
Cell 02840892: Defines a helper function load_and_preprocess_image to load and preprocess a single image. Also defines predict_image_pair for making a prediction on a single image pair without returning attention.
Cell 98169ea3: Defines the predict_labels helper function to get true labels, predicted labels, and raw logits for the entire test set using a trained model. Calls this function to get test set predictions and metrics.
Cell 686c8cb9: Re-defines load_and_preprocess_image and defines predict_image_pair_with_attention to predict the label for a single image pair and return attention weights and the true label.
Cell fe76825b: Demonstrates the usage of predict_image_pair_with_attention by displaying details (paths, labels, attention shapes) for the first two samples in sample_results.
Cell a0de1f70: Defines visualize_attention_map to overlay attention heatmaps on images and visualize_gradcam_heatmap to overlay Grad-CAM heatmaps on images.
Cell c577bdb3: Attempts to visualize attention and Grad-CAM heatmaps for sample images, but encountered a KeyError, indicating that 'ct_heatmap' and 'mri_heatmap' were not successfully added to sample_results. This cell needs debugging.
Cell bvm2CXvOOBld: Defines visualize_attention_maps_side_by_side to visualize CT→MRI and MRI→CT attention maps overlaid on their respective images side by side.
Cell c2d6569c: Defines the GradCAM class for generating Grad-CAM heatmaps.
Cell 6495b786: Loads the trained model and uses the predict_labels function to get the true labels, predicted labels, and logits for the test set. Prints these details for the first few samples.
Cell az4LU9rVuNlJ: Initializes GradCAM instances for the CT and MRI encoders. Uses select_sample_images_for_visualization (re-defined implicitly here or assuming it was run) to get sample details. Iterates through samples, generates Grad-CAM heatmaps using the GradCAM instances, and attempts to visualize attention and Grad-CAM heatmaps using the visualization functions. This cell explicitly calls visualize_attention_map and visualize_gradcam_heatmap.
Cell rbe2umzZW3pS: Markdown cell introducing the visualization of attention maps side by side.
Cell 1G5Afn6mS2Hq: Calls the visualize_attention_maps_side_by_side function for a specific sample image pair using previously obtained attention weights (ct_mri_attn, mri_ct_attn).
Cell 1jpXwkTxMd18: Markdown cell introducing Feature Maps visualization.
Cell -G_8H3VsMiSo: Code cell to set up a forward hook on the last convolutional layer of the CT CNN encoder to capture its feature maps for a sample image. Performs a forward pass and prints the shape of the captured feature maps.
Cell 2Ndso1StNfeQ: Defines visualize_spatial_fused_heatmap to visualize the spatial fused features from the model's intermediate layer as a heatmap overlaid on an image. Calls this function for the sample image pair using the spatial_fused_features obtained from a forward pass through the trained model.
Cell 8Jut8f57OMF8: Visualizes the feature maps captured from the CT CNN encoder in cell -G_8H3VsMiSo.
Cell xEUrJw70HHg2: Defines helper functions cosine_similarity_matrix and compute_and_plot_similarity for calculating and visualizing the cosine similarity matrix between CT and MRI patch embeddings before and after the cross-attention fusion. This cell contains a SyntaxError and will fail.
Cell b76d86e8: Markdown cell introducing the calculation of evaluation metrics.
Cell 500c80ce: Uses the predict_labels function to get true and predicted labels for the test set. Calculates and prints accuracy, confusion matrix, precision, recall, and F1-score based on these labels.
Cell f795d468: Defines the calculate_ROC function to compute the true labels and predicted probabilities for the positive class from the test set.
Cell aP7ZXyW7ThcM: Calls calculate_ROC to get true labels and probabilities. Calculates the ROC curve and AUC score using roc_curve and auc from sklearn.metrics. Plots the ROC curve and prints the AUC score.
Cell 9281e1ef: Markdown cell describing the process of generating predictions, attention weights, and Grad-CAM heatmaps for sample images.
Cell 9a9ceb8d: Re-defines the GradCAM class (redundant).
Cell PG_jNKu3XGPO: Calculates and prints the number of trainable parameters in the trained_model.
Cell 562DQRD9XPxm: Loads the trained model again. Iterates through sample image paths (assuming sample_ct_image_paths_viz and sample_mri_image_paths_viz are populated). Uses predict_image_pair_with_attention to get attention weights and calculates and prints detailed statistics (mean, max, min per head and overall) for CT→MRI and MRI→CT attention for each sample. Stores these statistics in sample_attention_stats.
Cell scMAaaC6XXjF: Markdown cell introducing Cross-Attention Statistics.
Cell df718584: Uses the collected sample_attention_stats and sample_true_labels_viz (assuming it's available) to group attention statistics by true class label and calculate the average statistics (mean, max, min) for each class. Prints these average class statistics.
Cell 1FBfTMyYFf-o: Markdown cell indicating the start of the "Attention maps" subtask.
Cell eynEgJ-iEg7N: Markdown cell indicating "Attention sketch maps and heatmaps".
Cell 6dd2aa79: Defines the visualize_attention_sketch_map function to visualize attention using lines connecting query patches to their top-K attended key patches, overlaid on the original image.
Cell QPlifrD9q-Tq: Re-defines the visualize_attention_sketch_map function with slight modifications (e.g., blending with transparency, focusing on top queries). This is a redundant definition.
Cell a1c5031a: Iterates through sample_results and calls the (second) visualize_attention_sketch_map function to generate and display sketch maps for both CT→MRI and MRI→CT attention for each sample.
Cell 3bx_u1yIsNHX: Defines visualize_attention_heatmap to visualize attention weights as a standard heatmap overlay on the original image.
Cell 0v-EABMJtaBU: Iterates through sample_results and calls the visualize_attention_heatmap function to generate and display attention heatmaps for both CT→MRI and MRI→CT attention for each sample.
