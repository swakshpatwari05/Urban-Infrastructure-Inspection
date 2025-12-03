import streamlit as st
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from skimage.util import img_as_float
from skimage import measure
from PIL import Image
import io
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import stats
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wall Crack Detection", layout="wide")
st.title("🧱 Wall Detection & Segmentation")

# Create tabs for different methods
tab1, tab2 = st.tabs(["Traditional CV Approach", "Deep Learning Approach"])

@st.cache_data
def load_image(image_file):
    # Convert uploaded file to a cv2 image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img

def process_image_traditional(img):
    # Make copies for later overlays
    img_orig = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Harris Corner Detection
    # -------------------------------
    harris = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    harris_img = img.copy()
    harris_img[harris > 0.01 * harris.max()] = [255, 0, 0]  # Mark corners in red
    
    # Create binary mask for harris corners for comparison
    harris_mask = np.zeros_like(gray, dtype=np.uint8)
    harris_mask[harris > 0.01 * harris.max()] = 255

    # -------------------------------
    # Canny + Morphological Closing
    # -------------------------------
    edges = cv2.Canny(gray, 50, 120)
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # -------------------------------
    # HoughLinesP Detection
    # -------------------------------
    hough_img = img.copy()
    hough_mask = np.zeros_like(gray, dtype=np.uint8)
    lines = cv2.HoughLinesP(closed_edges, rho=1, theta=np.pi/180, 
                            threshold=50, minLineLength=28, maxLineGap=15)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)

    # -------------------------------
    # Region-based Segmentation (Contours)
    # -------------------------------
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region_seg_img = img.copy()
    region_mask = np.zeros_like(gray, dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(region_seg_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(region_mask, (x, y), (x + w, y + h), 255, 2)

    # -------------------------------
    # Crack Detection Logic (Hough + Region)
    # -------------------------------
    hough_detected = lines is not None and len(lines) > 0
    region_detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # more stringent area threshold for detection
            region_detected = True
            break
    traditional_prediction = 1 if (hough_detected or region_detected) else 0

    return harris_img, closed_edges, hough_img, region_seg_img, harris_mask, hough_mask, region_mask, traditional_prediction

def process_image_slic(img):
    # Make a copy of the original image
    img_orig = img.copy()
    
    # First, perform the same basic processing for Hough detection:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    lines1 = cv2.HoughLinesP(closed_edges, rho=1, theta=np.pi/180,
                             threshold=50, minLineLength=28, maxLineGap=15)
    hough_detected = lines1 is not None and len(lines1) > 0

    # Now, perform SLIC segmentation on the original image.
    # Convert image to RGB float (as required by skimage)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)
    
    # Run SLIC with chosen parameters; adjust n_segments for granularity
    segments = slic(img_float, n_segments=200, compactness=10, sigma=1, start_label=1)
    
    # Mark boundaries for visualization
    slic_img = mark_boundaries(img_float, segments)
    
    # Extract boundaries from the SLIC segmentation
    boundaries = find_boundaries(segments, mode='outer').astype(np.uint8) * 255
    
    # Run Hough Transform on the boundary mask
    hough_lines_img = img.copy()
    slic_mask = np.zeros_like(gray, dtype=np.uint8)
    lines2 = cv2.HoughLinesP(boundaries, rho=1, theta=np.pi/180,
                             threshold=50, minLineLength=28, maxLineGap=15)
    
    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(slic_mask, (x1, y1), (x2, y2), 255, 2)
    
    slic_detected = lines2 is not None and len(lines2) > 0

    # Union logic: if either Hough on closed_edges or on SLIC boundaries detects a crack, return 1.
    prediction = 1 if (hough_detected or slic_detected) else 0
    
    return slic_img, hough_lines_img, slic_mask, prediction

# GMM segmentation function
def apply_gmm_binary_mask(image, k=2):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    pixels = gray.reshape(-1, 1)
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)

    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(pixels_scaled)
    labels_image = labels.reshape(gray.shape)

    class_means = [pixels[labels == i].mean() for i in range(k)]
    crack_class = np.argmin(class_means)

    binary_mask = (labels_image == crack_class).astype(np.uint8) * 255
    return image_rgb, binary_mask

# Function to get DeepLabV3 model
@st.cache_resource
def load_dl_model():
    # Initialize the DeepLabV3 model
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    # Modify the classifier to output 1 class (binary segmentation)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    
    # Load pretrained weights from the local file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(os.path.dirname(__file__), "deepnetv3.pth")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        return model, device, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return model, device, False

# Function to preprocess image for deep learning
def preprocess_image_dl(img):
    # Resize image to 512x512 (or the size expected by your model)
    img_resized = cv2.resize(img, (512, 512))
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    # Add batch dimension
    img_batch = img_tensor.unsqueeze(0)
    return img_batch, img_rgb

# Function to predict mask using the pretrained model
def predict_mask(model, img_tensor, device, threshold=0.5):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)['out']
        pred = torch.sigmoid(output)
        pred_bin = (pred > threshold).float()
    return pred_bin

# ------------------- Refinement Methods -------------------

# 1. SLIC-based Refinement
def generate_superpixels(image, n_segments=300, compactness=10):
    # image must be in HWC format and float
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    superpixels = slic(image, n_segments=n_segments, compactness=compactness, start_label=0)
    return superpixels

def refine_with_superpixels(pred_mask, superpixels):
    refined_mask = np.zeros_like(pred_mask)

    for label in np.unique(superpixels):
        region_mask = (superpixels == label)
        majority_vote = stats.mode(pred_mask[region_mask].flatten(), keepdims=False).mode
        refined_mask[region_mask] = majority_vote

    return refined_mask

# 2. Region-based Split and Merge Refinement
def region_based_split_and_merge(image, pred_mask, split_variance_thresh=0.01, merge_similarity_thresh=0.1, min_region_size=50):
    image_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    labeled_mask = label(pred_mask)

    region_map = np.zeros_like(pred_mask)

    # Region Splitting
    label_counter = 1
    for region in regionprops(labeled_mask):
        coords = region.coords
        region_intensity = image_gray[tuple(zip(*coords))]
        region_variance = np.var(region_intensity)

        if region_variance > split_variance_thresh and len(coords) > 10:
            # split into 2 by k-means clustering
            pixels = np.array([image[coord[0], coord[1]] for coord in coords])
            pixels = np.float32(pixels)
            
            if len(pixels) > 1:  # Ensure there are enough pixels for clustering
                try:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, _ = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
                    labels = labels.flatten()
                    for i, coord in enumerate(coords):
                        region_map[coord[0], coord[1]] = label_counter + labels[i]
                    label_counter += 2
                except:
                    # Fallback if kmeans fails
                    for coord in coords:
                        region_map[coord[0], coord[1]] = label_counter
                    label_counter += 1
            else:
                for coord in coords:
                    region_map[coord[0], coord[1]] = label_counter
                label_counter += 1
        else:
            for coord in coords:
                region_map[coord[0], coord[1]] = label_counter
            label_counter += 1

    # Region Merging
    final_map = region_map.copy()
    merged = set()
    for label1 in np.unique(region_map):
        if label1 == 0 or label1 in merged:
            continue

        mask1 = (region_map == label1)
        if np.sum(mask1) == 0:  # Skip empty regions
            continue
            
        mean1 = np.mean(image[mask1], axis=0)

        for label2 in np.unique(region_map):
            if label2 == 0 or label2 == label1 or label2 in merged:
                continue

            mask2 = (region_map == label2)
            if np.sum(mask2) == 0:  # Skip empty regions
                continue
                
            mean2 = np.mean(image[mask2], axis=0)

            # Merge if similar color mean
            if np.linalg.norm(mean1 - mean2) < merge_similarity_thresh:
                final_map[mask2] = label1
                merged.add(label2)

    # Morphological Cleaning
    binary_mask = final_map > 0
    binary_mask = binary_fill_holes(binary_mask)
    binary_mask = remove_small_objects(binary_mask, min_size=min_region_size)
    binary_mask = remove_small_holes(binary_mask, area_threshold=min_region_size)

    return binary_mask.astype(np.uint8)

# 3. Mean Shift Refinement
def generate_mean_shift_segments(image, spatial_radius=21, color_radius=51, quantization_level=16):
    # Convert image to uint8 [0,255] and then to BGR
    image_uint8 = (image * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    
    # Apply mean shift filtering
    filtered_bgr = cv2.pyrMeanShiftFiltering(image_bgr, spatial_radius, color_radius)
    filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
    
    # Quantize colors
    quantized = (filtered_rgb // quantization_level) * quantization_level
    
    # Flatten to combine channels; cast to uint32 to avoid overflow
    flat = quantized.reshape(-1, 3).astype(np.uint32)
    flat_int = flat[:, 0] * 256 * 256 + flat[:, 1] * 256 + flat[:, 2]
    quantized_int = flat_int.reshape(quantized.shape[0], quantized.shape[1])
    
    # Label connected components
    segments = label(quantized_int, connectivity=1)
    return segments

def refine_with_mean_shift(pred_mask, segments):
    refined_mask = np.zeros_like(pred_mask)
    unique_segments = np.unique(segments)
    
    for seg_val in unique_segments:
        region = (segments == seg_val)
        majority_label = np.mean(pred_mask[region]) > 0.5
        refined_mask[region] = majority_label
        
    return refined_mask

# Function to visualize the prediction with refinements
def visualize_prediction_with_refinements(image_tensor, pred_bin, slic_refined, region_refined, mean_shift_refined, alpha=0.6):
    # Convert tensors to numpy arrays
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
    pred_np = pred_bin.squeeze().cpu().numpy()
    
    # Create overlays
    overlay_original = create_overlay(image_np, pred_np, alpha)
    overlay_slic = create_overlay(image_np, slic_refined, alpha)
    overlay_region = create_overlay(image_np, region_refined, alpha)
    overlay_mean_shift = create_overlay(image_np, mean_shift_refined, alpha)
    
    # Create figure for visualization
    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    
    # Original Image and Predicted Mask
    axs[0, 0].imshow(image_np)
    axs[0, 0].set_title('Original Image')
    axs[0, 1].imshow(pred_np, cmap='gray')
    axs[0, 1].set_title('Predicted Mask')
    
    # Refined Masks
    axs[0, 2].imshow(slic_refined, cmap='gray')
    axs[0, 2].set_title('SLIC Refined')
    axs[0, 3].imshow(region_refined, cmap='gray')
    axs[0, 3].set_title('Region-based Refined')
    
    # Mean Shift Refined and Overlays
    axs[1, 0].imshow(mean_shift_refined, cmap='gray')
    axs[1, 0].set_title('Mean Shift Refined')
    axs[1, 1].imshow(overlay_original)
    axs[1, 1].set_title('Original Overlay')
    axs[1, 2].imshow(overlay_slic)
    axs[1, 2].set_title('SLIC Overlay')
    axs[1, 3].imshow(overlay_mean_shift)
    axs[1, 3].set_title('Mean Shift Overlay')
    
    for ax in axs.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def create_overlay(image, mask, alpha=0.6):
    # Create green mask overlay
    overlay = image.copy()
    green_mask = np.zeros_like(image)
    green_mask[..., 1] = 1  # Green channel
    overlay_mask = np.where(mask[..., None] > 0, green_mask, 0)
    overlay = (1 - alpha) * image + alpha * overlay_mask
    overlay = np.clip(overlay, 0, 1)
    return overlay

# Function to visualize traditional CV methods comparison
def visualize_traditional_cv_comparison(img_rgb, harris_mask, hough_mask, region_mask, slic_mask, gmm_mask, alpha=0.6):
    # Create figure for visualization
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original Image
    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title('Original Image')
    
    # Harris Corners Mask
    axs[0, 1].imshow(harris_mask, cmap='gray')
    axs[0, 1].set_title('Harris Corners Mask')
    
    # Hough Lines Mask
    axs[0, 2].imshow(hough_mask, cmap='gray')
    axs[0, 2].set_title('Hough Lines Mask')
    
    # Region Mask
    axs[1, 0].imshow(region_mask, cmap='gray')
    axs[1, 0].set_title('Region-based Mask')
    
    # SLIC Mask
    axs[1, 1].imshow(slic_mask, cmap='gray')
    axs[1, 1].set_title('SLIC Mask')
    
    # GMM Mask
    axs[1, 2].imshow(gmm_mask, cmap='gray')
    axs[1, 2].set_title('GMM Mask')
    
    for ax in axs.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def visualize_traditional_overlays(img_rgb, harris_mask, hough_mask, region_mask, slic_mask, gmm_mask, alpha=0.6):
    # Create overlays for each mask
    harris_overlay = create_overlay(img_rgb, harris_mask/255, alpha)
    hough_overlay = create_overlay(img_rgb, hough_mask/255, alpha)
    region_overlay = create_overlay(img_rgb, region_mask/255, alpha)
    slic_overlay = create_overlay(img_rgb, slic_mask/255, alpha)
    gmm_overlay = create_overlay(img_rgb, gmm_mask/255, alpha)
    
    # Create figure for visualization
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original Image
    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title('Original Image')
    
    # Harris Corners Overlay
    axs[0, 1].imshow(harris_overlay)
    axs[0, 1].set_title('Harris Corners Overlay')
    
    # Hough Lines Overlay
    axs[0, 2].imshow(hough_overlay)
    axs[0, 2].set_title('Hough Lines Overlay')
    
    # Region Overlay
    axs[1, 0].imshow(region_overlay)
    axs[1, 0].set_title('Region-based Overlay')
    
    # SLIC Overlay
    axs[1, 1].imshow(slic_overlay)
    axs[1, 1].set_title('SLIC Overlay')
    
    # GMM Overlay
    axs[1, 2].imshow(gmm_overlay)
    axs[1, 2].set_title('GMM Overlay')
    
    for ax in axs.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return buf

# ------------- Streamlit UI -------------
with tab1:
    st.header("Traditional Computer Vision Approach")
    
    uploaded_file = st.file_uploader("Upload a wall image (JPG/PNG/JPEG):", type=["jpg","png","jpeg"], key="cv_uploader")
    
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        
        # Process the image through our traditional pipeline
        harris_img, closed_edges, hough_img, region_seg_img, harris_mask, hough_mask, region_mask, traditional_prediction = process_image_traditional(img)
        
        # Process the image using SLIC
        slic_img, slic_hough_img, slic_mask, slic_prediction = process_image_slic(img)
        
        # Process the image using GMM
        image_rgb, gmm_mask = apply_gmm_binary_mask(img, 2)
        
        # GMM Crack Detection Logic
        crack_pixel_ratio = np.sum(gmm_mask == 255) / gmm_mask.size
        gmm_prediction = 1 if crack_pixel_ratio > 0.01 else 0  # 1% threshold
        
        st.subheader("Uploaded Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        
        # Display traditional CV outputs
        st.subheader("Traditional Computer Vision Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB), caption="Harris Corner Detection", use_column_width=True)
            st.image(closed_edges, caption="Canny + Morphological Closing", use_column_width=True, channels="GRAY")
        
        with col2:
            st.image(cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB), caption="Crack Lines via HoughLinesP", use_column_width=True)
            st.image(cv2.cvtColor(region_seg_img, cv2.COLOR_BGR2RGB), caption="Region-based Segmentation", use_column_width=True)
        
        with col3:
            # Final Crack Classification
            if traditional_prediction == 1:
                st.error("⚠️ Crack Detected! (Traditional)")
            else:
                st.success("✅ No Crack Detected (Traditional)")
        
        # Display SLIC outputs
        st.subheader("SLIC-based Segmentation Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(slic_img, caption="SLIC Segmentation", use_column_width=True)
        
        with col2:
            # SLIC Crack Classification
            if slic_prediction == 1:
                st.error("⚠️ Crack Detected! (SLIC + Hough)")
            else:
                st.success("✅ No Crack Detected (SLIC + Hough)")

        # GMM Results
        st.subheader("GMM-Based Segmentation")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_rgb, caption="Original (RGB)", use_column_width=True)
        with col2:
            st.image(gmm_mask, caption="GMM Crack Mask", use_column_width=True, channels="GRAY")
            
        # Display GMM result in Streamlit
        if gmm_prediction == 1:
            st.error("⚠️ Crack Detected! (GMM Segmentation)")
        else:
            st.success("✅ No Crack Detected (GMM Segmentation)")

        # New: Detailed Refinement Results and Performance Comparison
        st.subheader("Detailed Method Comparison")
        
        # Visualization of all masks
        img_rgb_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_comparison = visualize_traditional_cv_comparison(
            img_rgb_viz, 
            harris_mask, 
            hough_mask, 
            region_mask, 
            slic_mask, 
            gmm_mask
        )
        st.image(mask_comparison, caption="Comparison of Detection Masks", use_column_width=True)
        
        
        # Performance Comparison
        st.subheader("Performance Comparison")
        
        # Calculate pixel coverage for each method
        harris_coverage = np.sum(harris_mask > 0) / harris_mask.size * 100
        hough_coverage = np.sum(hough_mask > 0) / hough_mask.size * 100
        region_coverage = np.sum(region_mask > 0) / region_mask.size * 100
        slic_coverage = np.sum(slic_mask > 0) / slic_mask.size * 100
        gmm_coverage = np.sum(gmm_mask > 0) / gmm_mask.size * 100
        
        data = {
            'Method': ['Harris Corners', 'Hough Lines', 'Region-based', 'SLIC', 'GMM'],
            'Crack Coverage (%)': [
                harris_coverage,
                hough_coverage,
                region_coverage,
                slic_coverage,
                gmm_coverage
            ]
        }
        
        st.write("Crack Coverage Percentage by Method:")
        st.bar_chart(data, x='Method', y='Crack Coverage (%)')
        
        # Method Analysis
        st.subheader("Method Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strongest Detection Methods**")
            methods = ['Harris Corners', 'Hough Lines', 'Region-based', 'SLIC', 'GMM']
            coverages = [harris_coverage, hough_coverage, region_coverage, slic_coverage, gmm_coverage]
            max_idx = coverages.index(max(coverages))
            
            st.info(f"**{methods[max_idx]}** shows the highest crack coverage at **{coverages[max_idx]:.2f}%**")
            
            if max(coverages) > 1.0:
                st.warning("Significant crack pattern detected in the image")
            else:
                st.success("Minimal crack patterns detected in the image")
        
        with col2:
            st.markdown("**Detection Confidence**")
            
            # Count positive detections
            detection_count = sum([
                1 if harris_coverage > 0.5 else 0,
                1 if hough_coverage > 0.5 else 0,
                1 if region_coverage > 0.5 else 0,
                1 if slic_coverage > 0.5 else 0,
                1 if gmm_coverage > 0.5 else 0
            ])
            confidence = (detection_count / 5) * 100
            
            st.progress(confidence / 100)
            st.write(f"Detection Confidence: {confidence:.1f}%")

            if confidence > 60:
                st.error("High confidence crack detection - further inspection recommended")
            elif confidence > 20:
                st.warning("Medium confidence crack detection - monitoring recommended")
            else:
                st.success("Low confidence crack detection - likely safe condition")

            # Final Analysis Summary
            st.subheader("Analysis Summary")
            st.markdown(f"""
            * **Traditional CV Methods**: {'Detected cracks' if traditional_prediction == 1 else 'No cracks detected'}
            * **SLIC-Based Methods**: {'Detected cracks' if slic_prediction == 1 else 'No cracks detected'}
            * **GMM-Based Methods**: {'Detected cracks' if gmm_prediction == 1 else 'No cracks detected'}
            * **Overall Detection Confidence**: {confidence:.1f}%
            """)

            # Add recommendations based on detection results
            if confidence > 40:
                st.error("""
                **Recommendations**: 
                - Consider professional inspection of the wall
                - Monitor crack development over time
                - Check for water damage or structural issues nearby
                """)
            elif confidence > 10:
                st.warning("""
                **Recommendations**:
                - Monitor the area periodically
                - Take reference photos for comparison over time
                - Check again after extreme weather conditions
                """)
            else:
                st.success("""
                **Recommendations**:
                - No immediate action required
                - Include in regular building maintenance checks
                """)

with tab2:
    st.header("Deep Learning Approach with Refinement Methods")
    
    uploaded_file_dl = st.file_uploader("Upload a wall image (JPG/PNG/JPEG):", type=["jpg","png","jpeg"], key="dl_uploader")
    
    # Load the pretrained model
    model, device, model_loaded = load_dl_model()
    
    if not model_loaded:
        st.warning("Pretrained model could not be loaded. Deep learning analysis may not be accurate.")
    
    if uploaded_file_dl is not None:
        with st.spinner("Processing image with DeepLabV3 model and applying refinements..."):
            img_dl = load_image(uploaded_file_dl)
            
            # Preprocess image
            img_tensor, img_rgb = preprocess_image_dl(img_dl)
            
            # Predict mask
            pred_bin = predict_mask(model, img_tensor, device)
            pred_np = pred_bin.squeeze().cpu().numpy()
            
            # Apply SLIC-based refinement
            superpixels = generate_superpixels(img_rgb, n_segments=300, compactness=10)
            slic_refined = refine_with_superpixels(pred_np, superpixels)
            
            # Apply Region-based Split and Merge refinement
            region_refined = region_based_split_and_merge(img_rgb, pred_np)
            
            # Apply Mean Shift refinement
            mean_shift_segments = generate_mean_shift_segments(img_rgb)
            mean_shift_refined = refine_with_mean_shift(pred_np, mean_shift_segments)
            
            # Visualize all results
            vis_buf = visualize_prediction_with_refinements(
                img_tensor.cpu(), 
                pred_bin.cpu(), 
                slic_refined, 
                region_refined, 
                mean_shift_refined
            )
            
            # Display results
            st.image(vis_buf, caption="DeepLabV3 Segmentation Results with Refinement Methods", use_column_width=True)
            
            # Show individual refinement results
            st.subheader("Detailed Refinement Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**SLIC-based Refinement**")
                st.info("Combines segmentation with majority voting inside superpixels")
                crack_ratio_slic = np.sum(slic_refined) / slic_refined.size
                if crack_ratio_slic > 0.01:
                    st.error("⚠️ Crack Detected! (SLIC Refinement)")
                else:
                    st.success("✅ No Crack Detected (SLIC Refinement)")
            
            with col2:
                st.markdown("**Region-based Refinement**")
                st.info("Uses split & merge based on color & texture variance")
                crack_ratio_region = np.sum(region_refined) / region_refined.size
                if crack_ratio_region > 0.01:
                    st.error("⚠️ Crack Detected! (Region Refinement)")
                else:
                    st.success("✅ No Crack Detected (Region Refinement)")
            
            with col3:
                st.markdown("**Mean Shift Refinement**")
                st.info("Uses adaptive bandwidth clustering for natural segmentation")
                crack_ratio_ms = np.sum(mean_shift_refined) / mean_shift_refined.size
                if crack_ratio_ms > 0.01:
                    st.error("⚠️ Crack Detected! (Mean Shift Refinement)")
                else:
                    st.success("✅ No Crack Detected (Mean Shift Refinement)")
            
            # Display comparison of methods
            st.subheader("Performance Comparison")
            
            # Calculate crack ratios for each method
            crack_ratio_original = np.sum(pred_np) / pred_np.size
            
            data = {
                'Method': ['Original DL', 'SLIC', 'Region-based', 'Mean Shift'],
                'Crack Coverage (%)': [
                    crack_ratio_original * 100,
                    crack_ratio_slic * 100, 
                    crack_ratio_region * 100,
                    crack_ratio_ms * 100
                ]
            }
            
            st.write("Crack Coverage Percentage by Method:")
            st.bar_chart(data, x='Method', y='Crack Coverage (%)')
            
            # Add new detailed analysis and recommendations section
            st.subheader("Deep Learning Model Analysis")
            
            # Calculate weighted ensemble prediction
            weights = {
                'Original': 0.25,
                'SLIC': 0.25,
                'Region': 0.25,
                'MeanShift': 0.25
            }
            
            ensemble_score = (
                weights['Original'] * crack_ratio_original +
                weights['SLIC'] * crack_ratio_slic +
                weights['Region'] * crack_ratio_region +
                weights['MeanShift'] * crack_ratio_ms
            ) * 100
            
            st.write(f"Weighted Ensemble Score: {ensemble_score:.2f}%")
            
            # Display confidence gauge
            st.progress(min(ensemble_score/10, 1.0))  # Cap at 100%
            
            if ensemble_score > 5:
                st.error("⚠️ High confidence crack detection (Deep Learning)")
                st.markdown("""
                **Model Analysis**:
                - The deep learning model has detected significant crack patterns with high confidence
                - Multiple refinement methods confirm the detection
                - Detailed inspection is recommended
                """)
            elif ensemble_score > 1:
                st.warning("⚠️ Medium confidence crack detection (Deep Learning)")
                st.markdown("""
                **Model Analysis**:
                - The model has detected potential crack patterns with moderate confidence
                - Some refinement methods confirm the detection
                - Further monitoring is recommended
                """)
            else:
                st.success("✅ Low/No crack detection (Deep Learning)")
                st.markdown("""
                **Model Analysis**:
                - The model found minimal or no crack patterns
                - Refinement methods confirm the absence of significant cracks
                - The wall appears to be in good condition
                """)
            
            # Method comparison and analysis
            st.subheader("Method Effectiveness Analysis")
            
            # Determine the most sensitive method
            methods = ['Original DL', 'SLIC', 'Region-based', 'Mean Shift']
            ratios = [crack_ratio_original, crack_ratio_slic, crack_ratio_region, crack_ratio_ms]
            most_sensitive_idx = np.argmax(ratios)
            
            st.info(f"Most sensitive method: **{methods[most_sensitive_idx]}** with {ratios[most_sensitive_idx]*100:.2f}% coverage")
            
            # Calculate agreement between methods
            agreement_count = sum([1 for r in ratios if r > 0.01])
            agreement_percentage = (agreement_count / len(ratios)) * 100
            
            st.write(f"Method agreement: {agreement_percentage:.1f}% ({agreement_count}/{len(ratios)} methods agree)")
            
            # Model confidence explanation
            st.subheader("Confidence Explanation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **How confidence is calculated:**
                - Original model prediction (25%)
                - SLIC refinement results (25%)
                - Region-based refinement (25%) 
                - Mean Shift refinement (25%)
                """)
            
            with col2:
                st.markdown("""
                **Interpreting the results:**
                - >5%: Significant crack detected
                - 1-5%: Potential crack detected
                - <1%: No significant crack detected
                """)
            
            # Final recommendation section
            st.subheader("Final Recommendations")
            
            if ensemble_score > 5:
                st.error("""
                **Professional Assessment Recommended:**
                - Schedule a structural inspection
                - Document the crack pattern and location
                - Monitor for changes in size or pattern
                - Check for water infiltration or other damage
                """)
            elif ensemble_score > 1:
                st.warning("""
                **Monitoring Recommended:**
                - Take reference photos for future comparison
                - Check the area after extreme weather conditions
                - Monitor for growth or pattern changes
                - Consider applying crack sealant if stable
                """)
            else:
                st.success("""
                **No Immediate Action Required:**
                - Include in regular building maintenance inspections
                - Re-analyze if visible changes occur
                - Consider this area low priority for repairs
                """)