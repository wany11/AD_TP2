import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
nan = np.nan
imgtmp = loadmat("Indian_pines_corrected.mat")
img = np.float32(imgtmp['indian_pines_corrected'])
removed_bands = list(range(103, 108)) + list(range(149, 164))
img = np.delete(img, removed_bands, axis=2)
maptmp = loadmat("Indian_pines_gt.mat")
map = maptmp['indian_pines_gt']

res = img[:, :, 18]
plt.imshow((res - np.min(res)) / (np.max(res) - np.min(res)))
plt.title("Component 18 of the image")
plt.show()

plt.imshow(map)
plt.title("Ground truth map")
plt.show()

img_reshaped = img.reshape(-1, img.shape[2])
pca = PCA()
img_pca = pca.fit_transform(img_reshaped)

explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.show()

principal_axis_projection = img_pca[:, 0].reshape(img.shape[0], img.shape[1])
plt.imshow(principal_axis_projection, cmap='gray')
plt.title('Projection on the Principal Axis')
plt.show()

color_projection = img_pca[:, :3].reshape(img.shape[0], img.shape[1], 3)
plt.imshow((color_projection - np.min(color_projection)) / (np.max(color_projection) - np.min(color_projection)))
plt.title('Projection on the First Three Principal Axes')
plt.show()


plt.imshow(map)
plt.title("Ground truth map")
plt.show()

padded_components_list = []
for k in range(3):
    padded_components = np.full(imgtmp['indian_pines_corrected'].shape[2], np.nan)
    valid_indices = [i for i in range(imgtmp['indian_pines_corrected'].shape[2]) if i not in removed_bands]
    padded_components[valid_indices] = pca.components_[k, :]
    padded_components_list.append(padded_components)
plt.figure(figsize=(10, 6))
for i, component in enumerate(padded_components_list):
    plt.plot(component, label=f'Principal Component {i+1}')
plt.title('Spectral Signatures of Principal Components')
plt.xlabel('Wavelength Index')
plt.ylabel('Component Value')
plt.legend()
plt.show()
