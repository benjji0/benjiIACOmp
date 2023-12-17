import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

print(plt.style.available)
plt.style.use('seaborn-darkgrid')

import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from multiprocessing import Pool
import multiprocessing as mp

from time import time
from io import BytesIO
from sys import getsizeof


def main():
    st.set_page_config(layout='wide')
    st.markdown("<h1 style='text-align: center;'>Image Compression</h1>", unsafe_allow_html=True)
    st.write('This tool compresses your image in size, preserving the most important colors\
    using k-Means algorithm. It also shows you k most dominant colors in your image.')

    img_file_buffer = st.file_uploader('Upload an image', type=['png', 'jpg'])

    if img_file_buffer is not None:
        # Dictionary initialization
        compressed_images = {}
        compressed_bytes = {}  # byte representation of the ndarray images for downloading
        compressed_sizes = {}
        compressed_percents = {}
        centroids_dict = {}  # Dominant colors for every k in kMeans

        # Keeping the original image size for calculating the percentage benefit
        initial_size = getsizeof(img_file_buffer)

        # Converting an image to numpy array
        bytes_data = img_file_buffer.getvalue()
        image = plt.imread(img_file_buffer)  # .astype(np.uint8)
        image_shape = image.shape

        # For solving problems arising with RGBA (maybe)
        if image_shape[-1] == 4:
            image = image[:, :, :-1]
        image_shape = image.shape

        # Keeping name and type of the image
        file_name = (img_file_buffer.name).split('.')
        name = '.'.join(file_name[:-1])
        image_type = file_name[-1]
        if image_type in ['JPG', 'jpg', 'JPEG']:
            image_type = 'jpeg'

        # Shape of the numpy array representing the image
        st.sidebar.header('Additional Information')
        st.sidebar.write('Image shape: ', image.shape)

        # Displaying the image
        st.write(f'Initial size: {initial_size / 1000} kilobytes.')
        st.image(image, caption='Original', clamp=True)

        # Image resizing and pixel normalization
        if image_type == 'jpeg':
            # So that the range of the pixels is [0,1]
            resized_image = image / 255
        elif image_type == 'png':
            # The range is already in [0,1] for png images so we don't need to divide
            resized_image = image

        # 3 dimension -> 2 dimension conversion for kMeans
        resized_image = resized_image.reshape(image.shape[0] * image.shape[1], image.shape[2])

        # Variables for showing the progress
        # progress_bar = st.progress(0.0)
        # percent_complete = 0
        # placeholder = st.empty()

        # Running the algorithm
        start = time()
        start_k = 3
        end_k = 13
        diff = end_k - start_k
        k = start_k

        # Getting ready a tuple for multiprocessing using starmap
        k_list = list(range(start_k, end_k))
        image_type_lst = [image_type] * diff
        resized_image_lst = [resized_image] * diff
        image_shape_lst = [image_shape] * diff
        args = (*zip(resized_image_lst, image_type_lst, image_shape_lst, k_list),)

        results = list(compress_helper(args))

        for compressed_image, byte_im, centroids in results:
            # percent_complete = (k - 1) / (end_k - 2)
            # compressed_image, byte_im = compress(resized_image, image_type, image_shape, k)
            current_size = getsizeof(byte_im)

            compressed_images[k] = compressed_image
            compressed_bytes[k] = byte_im
            compressed_sizes[k] = current_size / 1000
            compressed_percents[k] = f'{((current_size - initial_size) * 100) / initial_size:.2f}%'
            centroids_dict[k] = centroids

            # placeholder.text(f'Progress: {int(percent_complete * 100)}/100')
            k += 1

        end = time()
        st.write(f'The program executed in {end - start:.2f} seconds.')
        st.write(f'Current compression benefit in percents: {compressed_percents.get(end_k - 1)}')

        st.sidebar.write('Compressed size dictionary: ', compressed_percents)

        st.header('Original vs Compressed for different values of k')
        preferred_k = st.slider('Choose the k in kMeans: ', start_k, end_k - 1, end_k - 1)

        # Displaying the original and compressed images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4 style='text-align: center;'>Original Image</h1>", unsafe_allow_html=True)
            st.image(image, clamp=True)

        with col2:
            st.markdown("<h4 style='text-align: center;'>Compressed Image</h1>", unsafe_allow_html=True)
            st.image(compressed_images[preferred_k], clamp=True)

        # Download button for user
        btn = st.download_button(
            label='Download the image',
            data=compressed_bytes.get(preferred_k),
            file_name=f'{name}_{preferred_k}.{image_type}',
            mime=f'image/{image_type}'
        )

        # Showing 3 most dominant colors
        current_centroids = centroids_dict[preferred_k]
        width, height = 100, 100

        color_1 = np.full((height, width, 3), current_centroids[0])
        color_2 = np.full((height, width, 3), current_centroids[1])
        color_3 = np.full((height, width, 3), current_centroids[2])

        st.markdown("<h3 style='text-align: center;'>3 dominant colors</h1>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(color_1)
        with col2:
            st.image(color_2)
        with col3:
            st.image(color_3)

        sizes = pd.Series(compressed_sizes)

        # Plotting the graph for different values of k
        st.write('')
        st.markdown("<h3 style='text-align: center;'>A graph of size change dynamics</h1>", unsafe_allow_html=True)
        plot_graph(sizes, initial_size, start_k, end_k)


@st.cache(show_spinner=False)
def compress_helper(args):
    """Helper of compress to handle caching problems arising from direct call of compress

    Args:
        args (tuple): arguments for compress (resized_image, image_type, ...)

    Returns:
        results (list): A list containing compressed images for different values of k

    """

    # For accessing all CPU cores
    num_workers = mp.cpu_count()

    # multiprocessing using Pool and starmap
    pool = mp.Pool(num_workers)
    results = pool.starmap(compress, iterable=args)

    pool.close()
    pool.join()

    return results


@st.cache(show_spinner=False)
def compress(resized_image, image_type, initial_image_shape, k):
    """Compresses an image using kMeans algorithm

    Args:
        resized_image (ndarray): The resized 2D representation of inital 3D image
        initial_image_shape (tuple): The shape of the inital image for recovery
        k (int): Number of clusters in kMeans

    Returns:
        compressed_image (ndarray): Compressed image of appropriate (initial) size
        byte_im (byteIO): Byte represenation of the compressed image

    """

    max_iters = 100
    resized_image = np.array(resized_image)
    kmeans = KMeans(n_clusters=k, max_iter=max_iters, n_init=1).fit(resized_image)
    idx = kmeans.predict(resized_image)
    centroids = kmeans.cluster_centers_

    # Representing the image in terms of indices
    compressed_image = centroids[idx, :]

    # Reshaping recovered image into proper dimensions and getting back to the proper shape
    compressed_image = compressed_image.reshape(initial_image_shape)
    compressed_image = (compressed_image * 255).astype(np.uint8)

    # Converting ndarray to image
    im = Image.fromarray(compressed_image, mode='RGB')

    # Converting image to bytes
    buf = BytesIO()
    im.save(buf, format=image_type)
    byte_im = buf.getvalue()
    print('kMeans K=', k)

    return compressed_image, byte_im, centroids


def plot_graph(sizes, initial_size, start_k, end_k):
    """Plots the graph of sizes per different numbers of clusters

    Args:
        sizes (float): Sizes of the byte representations of the compressed images
        initial_size (float): The byte size of the original image
        start_k (int): The starting value of clusters for plotting
        end_k (int): The ending value of cluster for plotting

    Returns:
        None

    """

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(sizes)
    ax.axhline(initial_size / 1000, color='red', label='initial size')
    ax.set_xlabel('k in kMeans')
    ax.set_ylabel('kilobytes')
    ax.set_title('Compressed image sizes')
    ax.set_xticks(range(start_k, end_k))

    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)


if __name__ == '__main__':
    main()