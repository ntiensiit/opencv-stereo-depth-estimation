import cv2
import numpy as np

FOCAL_LENGTH = 700.0
BASELINE = 0.12
MIN_DISPARITY = 1.0  # depth = f*B/d diverges when disparity is near zero

def load_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    result = image
    return result

def create_matcher():
    block = 5
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=block, P1=8 * block**2, P2=32 * block**2, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    result = matcher
    return result

def compute_disparity(left, right):
    matcher = create_matcher()
    raw = matcher.compute(left, right).astype(np.float32)
    disparity = raw / 16.0  # SGBM stores 4 fractional bits in fixed point
    result = disparity
    return result

def compute_depth(disparity, focal_length, baseline):
    valid = disparity >= MIN_DISPARITY
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = focal_length * baseline / disparity[valid]
    result = depth
    return result

def normalize_map(values):
    valid = np.isfinite(values)
    display = np.zeros(values.shape, dtype=np.uint8)
    display[valid] = cv2.normalize(values[valid], None, 0, 255, cv2.NORM_MINMAX).ravel()
    result = display
    return result

def scale_percentile(values, low, high):
    vmin, vmax = np.percentile(values, [low, high])
    if vmax <= vmin:
        result = np.full(values.shape, 128, dtype=np.uint8)
        return result
    normed = (np.clip(values, vmin, vmax) - vmin) / (vmax - vmin) * 255
    result = normed.astype(np.uint8)
    return result

def normalize_map_percentile(values, low=2, high=98):
    valid = np.isfinite(values)
    display = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        display[valid] = scale_percentile(values[valid], low, high)  # outlier-resistant depth display
    result = display
    return result

def colorize(values, normalize=normalize_map):
    normalized = normalize(values)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    result = colored
    return result

def save_maps(disparity, depth):
    cv2.imwrite("disparity.png", colorize(disparity))
    cv2.imwrite("depth.png", colorize(depth, normalize=normalize_map_percentile))

def main():
    left = load_gray("aloeL.jpg")
    right = load_gray("aloeR.jpg")
    disparity = compute_disparity(left, right)
    depth = compute_depth(disparity, FOCAL_LENGTH, BASELINE)
    save_maps(disparity, depth)

if __name__ == "__main__":
    main()
