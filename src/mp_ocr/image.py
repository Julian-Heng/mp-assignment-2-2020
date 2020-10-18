#!/usr/bin/env python3

""" Image module """

from pathlib import Path

import cv2


class Image:
    """Image class to hold attributes about an image

    Attributes
    ----------
    _filepath : str
        The filepath to the image file
    _image : ndarray
        The image as read by opencv imread method
    filepath
    filename
    filename_without_extension
    extension
    parentname
    image
    resolution
    height
    width
    area
    """

    def __init__(self, filepath):
        """Image constructor

        Parameters
        ----------
        filepath : str
            The filepath to the image file

        Returns
        -------
        Image
            The constructed image

        Raises
        ------
        ValueError
            Raised when the image path given does not exist
        """
        self._filepath = Path(filepath)

        # Throw error if image file does not exist
        if not self._filepath.exists():
            raise ValueError(f"Input image '{filepath}' does not exist")

        self._image = cv2.imread(str(self._filepath))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Image({self.filepath})"

    @property
    def filepath(self):
        """Image filepath

        Returns
        -------
        Path

        Examples
        --------
        >>> img = Image("/path/to/file.png")
        >>> img.filepath
        PoxixPath('/path/to/file.png')
        """
        return self._filepath

    @property
    def filename(self):
        """Image filename

        Returns
        -------
        str

        Examples
        --------
        >>> img = Image("/path/to/file.png")
        >>> img.filename
        'file.png'
        """
        return self._filepath.name

    @property
    def filename_without_extension(self):
        """Image filename without the file extension

        Returns
        -------
        str

        Examples
        --------
        >>> img = Image("/path/to/file.png")
        >>> img.filename_without_extension
        'file'
        """
        return self._filepath.stem

    @property
    def extension(self):
        """Image extension

        Returns
        -------
        str

        Examples
        --------
        >>> img = Image("/path/to/file.png")
        >>> img.extension
        '.png'
        """
        return self._filepath.suffix

    @property
    def parentpath(self):
        """Image parent path

        Returns
        -------
        Path

        Examples
        --------
        >>> img = Image("/path/to/file.png")
        >>> img.parentpath
        PoxixPath('/path/to/')
        """
        return self._filepath.parent

    @property
    def parentname(self):
        """Image parent directory name

        Returns
        -------
        str

        Examples
        --------
        >>> img = Image("/path/to/file.png")
        >>> img.parentname
        'to'
        """
        return self.parentpath.name

    @property
    def image(self):
        """Image array

        Returns
        -------
        ndarray

        Notes
        -----
        This returns a copy of the image
        """
        return self._image.copy()

    @property
    def resolution(self):
        """Image resolution

        Returns
        -------
        tuple
            The resolution of the image in the form (h, w)
        """
        return self._image.shape[:-1]

    @property
    def height(self):
        """Image height

        Returns
        -------
        int
        """
        return self.resolution[0]

    @property
    def width(self):
        """Image width

        Returns
        -------
        int
        """
        return self.resolution[1]

    @property
    def area(self):
        """Image area

        Returns
        -------
        int
        """
        return self.height * self.width
