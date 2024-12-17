import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
import scipy.ndimage as ndi
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from PIL import Image, ImageDraw, ImageFont

from .core.PSF import MicroscopeParameters

labelfont = ImageFont.load_default_imagefont()


def interactive_imshow_3d(image_3d, cmap="gray", initial_slice=None, **kwargs):
    """Displays an interactive matplotlib window with a slider to navigate
    through z-planes of a 3D image.

    Parameters:
    - image_3d: numpy.ndarray
        3D image data with shape (Nz, Ny, Nx).
    - cmap: str or Colormap, optional
        Colormap to use for displaying the image. Default is 'gray'.
    - initial_slice: int, optional
        The initial z-slice to display. Defaults to the middle slice.

    """
    Nz, Ny, Nx = image_3d.shape

    if initial_slice is None:
        initial_slice = Nz // 2
    else:
        if not (0 <= initial_slice < Nz):
            raise ValueError(f"initial_slice must be within [0, {Nz-1}]")

    # Create the figure and the initial image plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.25, bottom=0.25)  # Make space for the slider

    # Display the initial slice
    img_display = ax.imshow(image_3d[initial_slice, :, :], cmap=cmap, **kwargs)
    ax.set_title(f"Slice {initial_slice + 1}/{Nz}")
    ax.axis("off")  # Hide axes for better visualization

    # Create the slider axis
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgray")

    # Create the slider
    slider = Slider(
        ax=ax_slider,
        label="Z Slice",
        valmin=0,
        valmax=Nz - 1,
        valinit=initial_slice,
        valfmt="%0.0f",
        valstep=1,  # Ensure slider moves in integer steps
    )

    # Update function to be called when slider value changes
    def update(val):
        slice_idx = int(slider.val)
        img_display.set_data(image_3d[slice_idx, :, :])
        ax.set_title(f"Slice {slice_idx + 1}/{Nz}")
        fig.canvas.draw_idle()

    # Connect the update function to the slider
    slider.on_changed(update)

    plt.show()


class SimpleOrthoViewer:
    def __init__(
        self, image_3d, cmap="gray", norm=None, lateral_to_axial_ratio=0.333
    ):
        """
        Initialize the Simple Ortho Viewer with a 3D image.

        Parameters:
        - image_3d: numpy.ndarray
            3D image data with shape (Nz, Ny, Nx).
        - cmap: str or Colormap, optional
            Colormap to use for displaying the images.
        - lateral_to_axial_ratio: float, optional
        """
        self.image = image_3d
        self.cmap = cmap
        self.Nz, self.Ny, self.Nx = self.image.shape
        self.dyx_dz_ratio = lateral_to_axial_ratio
        # Initialize slice indices at the center
        self.z = self.Nz // 2
        self.y = self.Ny // 2
        self.x = self.Nx // 2

        if norm is None:
            # compute percentile
            clo, chi = np.percentile(image_3d, (1.0, 99.0))
            norm = plt.Normalize(vmin=clo, vmax=chi)

        # Create the figure and layout using GridSpec
        _base_size = 6.75
        self.fig = plt.figure(
            figsize=(_base_size, _base_size * 1.05), constrained_layout=True
        )
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.gs = GridSpec(
            2,
            2,
            figure=self.fig,
            width_ratios=[1, self.Nz / self.Nx / self.dyx_dz_ratio],
            height_ratios=[1, self.Nz / self.Ny / self.dyx_dz_ratio],
            wspace=0.01,
            hspace=0.01,
            left=0.1,
            right=0.9,
            bottom=0.4,
            top=0.9,
        )

        # Create axes for each orthogonal view
        self.ax_axial = self.fig.add_subplot(self.gs[0, 0])  # Top Left: XY
        # YZ
        self.ax_sagittal = self.fig.add_subplot(
            self.gs[0, 1],
            sharey=self.ax_axial,
        )
        # XZ
        self.ax_coronal = self.fig.add_subplot(
            self.gs[1, 0], sharex=self.ax_axial
        )

        # Display the initial slices
        self.im_axial = self.ax_axial.imshow(
            self.image[self.z, :, :],
            cmap=self.cmap,
            origin="lower",
            aspect="equal",
            norm=norm,
        )
        self.im_sagittal = self.ax_sagittal.imshow(
            self.image[:, :, self.x].T,
            cmap=self.cmap,
            origin="lower",
            aspect="equal",
            norm=norm,
        )  # Transposed for alignment
        self.im_coronal = self.ax_coronal.imshow(
            self.image[:, self.y, :],
            cmap=self.cmap,
            origin="lower",
            aspect="equal",
            norm=norm,
        )

        # Set titles
        self.ax_axial.set_title(f"(XY) Slice: Z={self.z}", fontsize=10)
        self.ax_sagittal.set_title(f"(YZ): X={self.x}", fontsize=10)
        self.ax_coronal.set_title(f"(XZ): Y={self.y}", fontsize=10)

        # Remove axis ticks for clarity
        for ax in [self.ax_axial, self.ax_sagittal, self.ax_coronal]:
            ax.axis("off")

        # Initialize crosshairs
        self.add_crosshairs()

        # Add colorbar to Axial view
        # self.add_colorbar()

        # Connect mouse scroll events to navigation functions
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

    def add_crosshairs(self):
        """
        Add crosshair lines to each orthogonal view.
        """
        # Axial (XY) view crosshairs: indicate X and Y positions
        self.cross_axial_v = self.ax_axial.axvline(
            self.x, color="r", linestyle="--", linewidth=0.8, alpha=0.25
        )
        self.cross_axial_h = self.ax_axial.axhline(
            self.y, color="r", linestyle="--", linewidth=0.8, alpha=0.25
        )

        # Sagittal (YZ) view crosshairs: indicate Y and Z positions
        self.cross_sagittal_v = self.ax_sagittal.axvline(
            self.z, color="r", linestyle="--", linewidth=0.8, alpha=0.25
        )
        self.cross_sagittal_h = self.ax_sagittal.axhline(
            self.y, color="r", linestyle="--", linewidth=0.8, alpha=0.25
        )

        # Coronal (XZ) view crosshairs: indicate X and Z positions
        self.cross_coronal_v = self.ax_coronal.axvline(
            self.x, color="r", linestyle="--", linewidth=0.8, alpha=0.25
        )
        self.cross_coronal_h = self.ax_coronal.axhline(
            self.z, color="r", linestyle="--", linewidth=0.8, alpha=0.25
        )

    def on_scroll(self, event):
        """
        Handle mouse scroll events to navigate through slices.

        Parameters:
        - event: matplotlib.backend_bases.MouseEvent
            The mouse event containing information about the scroll.
        """
        # Determine which axes the scroll occurred in
        if event.inaxes == self.ax_axial:
            # Scroll in Axial view: navigate Z-axis
            if event.button == "up":
                self.z = min(self.z + 1, self.Nz - 1)
            elif event.button == "down":
                self.z = max(self.z - 1, 0)
            self.update_axial()
        elif event.inaxes == self.ax_sagittal:
            # Scroll in Sagittal view: navigate X-axis
            if event.button == "up":
                self.x = min(self.x + 1, self.Nx - 1)
            elif event.button == "down":
                self.x = max(self.x - 1, 0)
            self.update_sagittal()
        elif event.inaxes == self.ax_coronal:
            # Scroll in Coronal view: navigate Y-axis
            if event.button == "up":
                self.y = min(self.y + 1, self.Ny - 1)
            elif event.button == "down":
                self.y = max(self.y - 1, 0)
            self.update_coronal()

    def on_click(self, event):
        if event.inaxes == self.ax_axial:
            y_click = int(event.ydata)
            x_click = int(event.xdata)
            self.x = x_click
            self.y = y_click
            self.update_axial()
            self.update_sagittal()
            self.update_coronal()

        if event.inaxes == self.ax_sagittal:
            y_click = int(event.ydata)
            z_click = int(event.xdata)
            self.y = y_click
            self.z = z_click
            self.update_axial()
            self.update_sagittal()
            self.update_coronal()

        if event.inaxes == self.ax_coronal:
            x_click = int(event.xdata)
            z_click = int(event.ydata)
            self.x = x_click
            self.z = z_click
            self.update_axial()
            self.update_sagittal()
            self.update_coronal()

    def update_axial(self):
        """
        Update the Axial (XY) view and crosshairs when Z-axis slice changes.
        """
        self.im_axial.set_data(self.image[self.z, :, :])
        self.ax_axial.set_title(f"(XY) Slice: Z={self.z}")

        # Update crosshairs
        self.cross_sagittal_v.set_xdata([self.z, self.z])
        self.cross_coronal_h.set_ydata([self.z, self.z])

        self.fig.canvas.draw_idle()

    def update_sagittal(self):
        """
        Update the Sagittal (YZ) view and crosshairs when X-axis slice changes.
        """
        # Transpose the YZ slice for proper alignment
        self.im_sagittal.set_data(self.image[:, :, self.x].T)
        self.ax_sagittal.set_title(f"(YZ) Slice: X={self.x}")

        # Update crosshairs
        self.cross_axial_v.set_xdata([self.x, self.x])
        self.cross_coronal_v.set_xdata([self.x, self.x])

        self.fig.canvas.draw_idle()

    def update_coronal(self):
        """
        Update the Coronal (XZ) view and crosshairs when Y-axis slice changes.
        """
        self.im_coronal.set_data(self.image[:, self.y, :])
        self.ax_coronal.set_title(f"(XZ) Slice: Y={self.y}")

        # Update crosshairs
        self.cross_axial_h.set_ydata([self.y, self.y])
        self.cross_sagittal_h.set_ydata([self.y, self.y])

        self.fig.canvas.draw_idle()

    def show(self):
        """
        Display the Ortho Viewer.
        """
        plt.show()


def get_dark_rgb_colormaps():
    rcmap = mcolors.LinearSegmentedColormap.from_list(
        "RedCh", ["#000000", "#ff00ff"]
    )
    gcmap = mcolors.LinearSegmentedColormap.from_list(
        "GreenCh", ["#000000", "#00ff00"]
    )
    bcmap = mcolors.LinearSegmentedColormap.from_list(
        "BlueCh", ["#000000", "#00ffff"]
    )
    return rcmap, gcmap, bcmap


def imshow(X, lo=0.1, hi=99.8, gamma=0.45):
    ch_axis = np.argmin(X.shape)
    nchs = X.shape[ch_axis]

    fig, ax = plt.subplots(ncols=nchs, sharex=True, sharey=True)

    aax = ax.ravel()

    for i in range(nchs):
        aax[i].imshow(
            np.take(X, i, axis=ch_axis), norm=mcolors.PowerNorm(gamma=gamma)
        )
        aax[i].axis("off")
        aax[i].set_title(f"ch = {i}")

    fig.suptitle(f"image shape = {X.shape}")

    return aax


def find_boundaries(segmentation):
    # Find boundaries by comparing each pixel to its neighbors
    boundaries = np.zeros(segmentation.shape, dtype=bool)
    for shift in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        shifted = np.roll(segmentation, shift, axis=(0, 1))
        boundaries |= segmentation != shifted
    return boundaries


def highlight_rgb(
    rgb_image,
    highlight_mask,
    highlight_color=(1.0, 1.0, 0.0),
    highlight_alpha=0.3,
):
    """create an RGB overlay for input RGB image"""
    composite = rgb_image.astype(float)
    composite /= composite.max()

    mask = highlight_mask.astype(bool)
    colored_mask = np.zeros((*highlight_mask.shape, 3))
    colored_mask[mask] = highlight_color
    composite = (
        composite * (1 - highlight_alpha * mask[:, :, np.newaxis])
        + colored_mask * highlight_alpha * mask[:, :, np.newaxis]
    )

    return np.uint8(np.clip(composite, 0, 1) * 255)


def _hex_to_rgb(hexstr):
    h = hexstr.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def assign_4colors(labeled_mask):
    """use 4-color theorem to assign an index 0-3 for each label"""

    unique_labels = np.unique(labeled_mask[labeled_mask != 0])

    color_assignments = {}

    for label in unique_labels:
        dilated = ndi.binary_dilation(labeled_mask == label)
        neighbors = set(labeled_mask[dilated]) - {label, 0}

        # find available colors
        used_colors = {color_assignments.get(n) for n in neighbors}
        available_colors = set(range(1, 7)) - used_colors

        color_assignments[label] = min(available_colors)

    return color_assignments


def colorize_segmentation(
    labels,
    color_list=[
        "#f0f2f5",
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
    ],
    label=True,
):
    """colorize segmentation using the 4-color theorem

    the first color in 'color_list' is used for background

    """
    assert len(color_list) == 7, "color_list must have 7 colors"

    color_indices = assign_4colors(labels)

    max_label = labels.max()
    # create a lookup rgb table
    color_lookup = np.zeros((max_label + 1, 3))
    # prepend background color for index-0
    rgb_list = [_hex_to_rgb(h) for h in color_list]
    color_lookup[0, :] = rgb_list[0]

    # map label id to rgb tuple
    for label, color_index in color_indices.items():
        color_lookup[label, :] = rgb_list[color_index]

    colored_labels = color_lookup[labels]
    colored_labels = np.uint8(colored_labels * 255)

    if label:
        # draw text on image
        ulabels = np.unique(labels)[1:]
        centers = ndi.center_of_mass(labels > 0, label, ulabels)
        cprops = dict(zip(ulabels, centers))
        pil_im = Image.fromarray(colored_labels)
        d = ImageDraw.Draw(pil_im)

        for label, cxy in cprops.items():
            d.text(cxy, f"{label}", font=labelfont, fill=(0, 0, 0))

        colored_labels_ann = np.array(pil_im)

        return colored_labels_ann

    return colored_labels


def split_rgb_image(X):
    ch_axis = np.argmin(X.shape)
    nch = X.shape[ch_axis]
    return [np.take(X, i, axis=ch_axis) for i in range(nch)]


def create_composite_image(image, colormaps, lo=0.1, hi=99.9, gamma=0.45):
    """create a composite image from a multichannel 2d image

    Args
    ----
    image (list of arrays): list of 2-d arrays
    colormaps (list of colormaps): a colormap is either a string or a
        matplotlib colormap
    lo (float): percentile for low-intensity scaling
    hi (float): percentile for high-intensity scaling
    gamma (float): gamma correction, must be in the range of 0 to 1

    Returns
    -------
    RGB image (uint8)

    """
    images = split_rgb_image(image)

    assert len(images) == len(
        colormaps
    ), "number of images must match the number of colormaps"

    # Initialize an empty RGB image
    composite = np.zeros((*images[0].shape, 3))

    for img, cmap in zip(images, colormaps):
        # Normalize the image
        clo, chi = np.percentile(img, (lo, hi))
        inorm = mcolors.PowerNorm(gamma=gamma, vmin=clo, vmax=chi, clip=True)
        img_norm = inorm(img)

        # If cmap is a string, get the corresponding colormap from matplotlib
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        # Apply the colormap
        colored = cmap(img_norm)[:, :, :3]  # Exclude alpha channel

        # Add to the composite
        composite += colored

    # Normalize the composite
    return np.uint8(np.clip(composite, 0, 1) * 255)


def visualize_central_otf(
    otf: np.ndarray,
    microscope_parameters: MicroscopeParameters,
    ax: plt.Axes,
    **imshow_kwargs,
):
    Nz, Ny, Nx = otf.shape
    central_otf = np.abs(fft.fftshift(otf[0]))

    # compute frequency spacing
    fy = fft.fftshift(fft.fftfreq(Ny, d=microscope_parameters.pixel_size))
    fx = fft.fftshift(fft.fftfreq(Nx, d=microscope_parameters.pixel_size))

    ax.imshow(
        central_otf,
        **imshow_kwargs,
        extent=[fx.min(), fx.max(), fy.min(), fy.max()],
    )

    # superimpose band limit
    bandlimit = (
        microscope_parameters.numerical_aperture
        / microscope_parameters.emission_wavelength
    )

    bcircle = Circle(
        (0, 0),
        radius=2 * bandlimit,
        color="red",
        linestyle="dashed",
        linewidth=1.5,
        fill=None,
    )
    ax.text(
        2 * bandlimit * 0.5,
        2 * bandlimit * 0.5,
        "2NA/$\lambda_{em}$",
        color="red",
        fontsize=12,
    )
    ax.add_artist(bcircle)
    ax.set_xlabel("spatial frequency, $\mu m^{-1}$")
    ax.set_ylabel("spatial frequency, $\mu m^{-1}$")
