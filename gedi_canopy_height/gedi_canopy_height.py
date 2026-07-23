import logging
import os
import posixpath
from os import makedirs
from os.path import exists, dirname, join, expanduser, abspath
from shutil import move
from subprocess import run, CalledProcessError, TimeoutExpired
from time import perf_counter
from typing import List

import requests

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import rasters as rt
from rasters import RasterGeometry, Raster

from .download import download

GEDI_DOWNLOAD_DIRECTORY = join("~", "data", "gedi-canopy-height")

CANOPY_COLORMAP = LinearSegmentedColormap.from_list(
    name="canopy_height",
    colors=[
        # "#0000ff",
        "#000000",
        "#745d1a",
        "#e1dea2",
        "#45ff01",
        "#325e32"
    ]
)

class GEDICanopyHeight:
    AUS_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_AUS.tif"
    NAFR_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_NAFR.tif"
    NAM_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_NAM.tif"
    NASIA_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_NASIA.tif"
    SAFR_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_SAFR.tif"
    SAM_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_SAM.tif"
    SASIA_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_SASIA.tif"

    logger = logging.getLogger(__name__)

    def __init__(self, source_directory: str = GEDI_DOWNLOAD_DIRECTORY):
        self.source_directory = source_directory

    def __repr__(self) -> str:
        return f'GEDICanopyHeight(source_directory="{self.source_directory}")'

    @staticmethod
    def _remote_file_size(URL: str, timeout: tuple[float, float] = (10.0, 30.0)) -> int | None:
        try:
            response = requests.head(URL, allow_redirects=True, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            return None

        content_length = response.headers.get("Content-Length")

        if content_length is None:
            return None

        try:
            return int(content_length)
        except ValueError:
            return None

    @staticmethod
    def _is_geotiff_readable(filename: str) -> bool:
        try:
            completed = run(
                ["gdalinfo", filename],
                check=False,
                capture_output=True,
                text=True,
                timeout=45,
            )
        except (OSError, TimeoutExpired):
            return False

        return completed.returncode == 0

    def _is_valid_source_file(self, URL: str, filename_absolute: str) -> bool:
        if not exists(filename_absolute):
            return False

        local_size = os.path.getsize(filename_absolute)
        if local_size <= 0:
            return False

        remote_size = self._remote_file_size(URL)
        if remote_size is not None and local_size != remote_size:
            self.logger.warning(
                "existing file size mismatch, expected %s but got %s: %s",
                remote_size,
                local_size,
                filename_absolute,
            )
            return False

        if not self._is_geotiff_readable(filename_absolute):
            self.logger.warning("existing file is not readable GeoTIFF: %s", filename_absolute)
            return False

        return True

    def _replace_invalid_file(self, filename_absolute: str) -> None:
        corrupt_filename = f"{filename_absolute}.corrupt"

        if exists(corrupt_filename):
            os.remove(corrupt_filename)

        move(filename_absolute, corrupt_filename)
        self.logger.warning("replaced corrupt file: %s -> %s", filename_absolute, corrupt_filename)

    def download_file(self, URL: str, filename: str) -> str:
        filename_absolute = abspath(expanduser(filename))

        if self._is_valid_source_file(URL=URL, filename_absolute=filename_absolute):
            self.logger.info(f"file already downloaded: {filename}")
            return filename

        for attempt in range(2):
            if exists(filename_absolute):
                self._replace_invalid_file(filename_absolute)

            self.logger.info(f"downloading: {URL} -> {filename}")
            directory = dirname(filename_absolute)
            makedirs(directory, exist_ok=True)
            partial_filename = f"{filename_absolute}.download"

            if attempt > 0 and exists(partial_filename):
                os.remove(partial_filename)

            if exists(partial_filename):
                partial_size = os.path.getsize(partial_filename)
                remote_size = self._remote_file_size(URL)
                if remote_size is not None and partial_size > remote_size:
                    os.remove(partial_filename)

            download_start = perf_counter()
            try:
                download(URL, partial_filename)
            except Exception:
                if attempt == 0:
                    self.logger.warning("retrying failed source download from scratch: %s", filename_absolute)
                    if exists(partial_filename):
                        os.remove(partial_filename)
                    continue
                raise

            download_end = perf_counter()
            download_duration = download_end - download_start
            self.logger.info(f"completed download in {download_duration:0.2f} seconds: {filename}")

            if not exists(partial_filename):
                raise IOError(f"unable to download URL: {URL}")

            move(partial_filename, filename_absolute)

            if self._is_valid_source_file(URL=URL, filename_absolute=filename_absolute):
                return filename

            self._replace_invalid_file(filename_absolute)

            if attempt == 0:
                self.logger.warning("retrying failed source download from scratch: %s", filename_absolute)

        raise IOError(f"downloaded file failed validation and was quarantined: {filename_absolute}")

    @property
    def source_URLs(self) -> dict:
        return {
            "AUS": self.AUS_URL,
            "NAFR": self.NAFR_URL,
            "NAM": self.NAM_URL,
            "NASIA": self.NASIA_URL,
            "SAFR": self.SAFR_URL,
            "SAM": self.SAM_URL,
            "SASIA": self.SASIA_URL
        }

    def source_URL(self, name: str) -> str:
        if name not in self.source_URLs:
            raise ValueError(f"unrecognized canopy height region: {name}")

        return self.source_URLs[name]

    def source_filename(self, name: str) -> str:
        URL = self.source_URL(name)
        filename_base = posixpath.basename(URL)
        filename = join(abspath(expanduser(self.source_directory)), filename_base)

        return filename

    def download_sources(self) -> List[str]:
        filenames = []

        for name, URL in self.source_URLs.items():
            filename = self.source_filename(name)
            self.download_file(URL, filename)

            if not exists(filename):
                raise IOError(f"unable to download {name} canopy height: {URL}")

            filenames.append(filename)

        return filenames

    @property
    def VRT_filename(self):
        return join(abspath(expanduser(self.source_directory)), "Forest_height_2019.vrt")

    @property
    def VRT(self) -> str:
        if exists(self.VRT_filename):
            try:
                completed = run(
                    ["gdalinfo", self.VRT_filename],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
            except TimeoutExpired:
                completed = None

            if completed is not None and completed.returncode == 0:
                return self.VRT_filename

            self.logger.warning("existing VRT is invalid or timed out and will be rebuilt: %s", self.VRT_filename)
            corrupt_vrt = f"{self.VRT_filename}.corrupt"
            if exists(corrupt_vrt):
                os.remove(corrupt_vrt)
            move(self.VRT_filename, corrupt_vrt)

        source_filenames = self.download_sources()
        command = ["gdalbuildvrt", self.VRT_filename, *source_filenames]
        self.logger.info(" ".join(command))

        try:
            run(command, check=True, capture_output=True, text=True, timeout=300)
        except FileNotFoundError as exc:
            raise RuntimeError("gdalbuildvrt is required but was not found in PATH") from exc
        except TimeoutExpired as exc:
            raise TimeoutError("gdalbuildvrt timed out while building canopy height VRT") from exc
        except CalledProcessError as exc:
            raise RuntimeError(f"gdalbuildvrt failed: {exc.stderr.strip()}") from exc

        if not exists(self.VRT_filename):
            raise IOError(f"unable to produce canopy height VRT: {self.VRT_filename}")
        
        return self.VRT_filename

    def canopy_height_meters(self, geometry: RasterGeometry, resampling=None) -> Raster:
        return rt.Raster.open(
            filename=self.VRT,
            geometry=geometry,
            nodata=np.nan,
            remove=101,
            resampling=resampling,
            cmap=CANOPY_COLORMAP
        )

def load_canopy_height(
        geometry: RasterGeometry, 
        resampling: str = "cubic", 
        source_directory: str = GEDI_DOWNLOAD_DIRECTORY) -> Raster:
    gedi = GEDICanopyHeight(source_directory=source_directory)
    canopy_height_meters = gedi.canopy_height_meters(geometry=geometry, resampling=resampling)

    return canopy_height_meters
