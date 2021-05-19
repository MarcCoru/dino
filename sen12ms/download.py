import os
import urllib
import zipfile
from tqdm import tqdm

H5URL = "https://syncandshare.lrz.de/dl/fiDJwH3ZgzcoDts3srTT8XaA/sen12ms.h5"
CSVURL = "https://syncandshare.lrz.de/dl/fiHr4oDKXzPSPYnPRWNxAqnk/sen12ms.csv"
REGIONSURL = "https://syncandshare.lrz.de/dl/fiELKg4TCSD9f57nfiGqys9R/regions.zip"
CSVSIZE = 47302099
H5SIZE = 115351475848

def download_sen12ms(root):
    h5file_path = os.path.join(root, "sen12ms.h5")
    paths_file = os.path.join(root, "sen12ms.csv")

    print(f"downloading {CSVURL} to {paths_file}")
    download_file(CSVURL, paths_file, overwrite=True)
    print(f"downloading {H5URL} to {h5file_path}")
    download_file(H5URL, h5file_path, overwrite=True)

def download_regions(root):
    regions_file = os.path.join(root, "regions.zip")
    download_file(REGIONSURL, regions_file, overwrite=True)
    unzip(regions_file, root)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path, overwrite=False):
    if url is None:
        raise ValueError("download_file: provided url is None!")

    if not os.path.exists(output_path) or overwrite:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        print(f"file exists in {output_path}. specify overwrite=True if intended")


def unzip(zipfile_path, target_dir):
    with zipfile.ZipFile(zipfile_path) as zip:
        for zip_info in zip.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, target_dir)