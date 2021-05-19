import torch
import os
import pandas as pd
from .data import trainregions, valregions, holdout_regions, data_transform
import h5py
import numpy as np
from .download import download_sen12ms, download_regions
import geopandas as gpd

class AllSen12MSDataset(torch.utils.data.Dataset):
    def __init__(self, root, fold, transform, tansform_coord=None,
                 classes=None, seasons=None, split_by_region=True, download=True):
        super(AllSen12MSDataset, self).__init__()

        self.transform = transform
        self.transform_coord = tansform_coord

        self.h5file_path = os.path.join(root, "sen12ms.h5")
        index_file = os.path.join(root, "sen12ms.csv")
        if not os.path.exists("regions.shp"):
            download_regions(".") # is tiny
        regions = gpd.read_file("regions.shp")[['region','geometry']].to_crs({'init': 'epsg:4326'})
        self.regions = {}
        for idx, row in regions.iterrows():
            lat, lon = row['geometry'].coords[0]
            self.regions[row['region']] = torch.tensor([(lon, lat)])


        if not os.path.exists(self.h5file_path) or not os.path.exists(index_file):
            if download:
                download_sen12ms(root)
            else:
                print(f"no dataset found at {root}. download via parameter download=True")
                import sys
                sys.exit()

        self.paths = pd.read_csv(index_file, index_col=0)

        if split_by_region:
            if fold == "train":
                regions = trainregions
            elif fold == "val":
                regions = valregions
            elif fold == "test":
                regions = holdout_regions
            elif fold == "all":
                regions = holdout_regions + valregions + trainregions
            else:
                raise AttributeError("one of meta_train, meta_val, meta_test must be true or "
                                     "fold must be in 'train','val','test'")

            mask = self.paths.region.isin(regions)
            print(f"fold {fold} specified. splitting by regions. Keeping {mask.sum()} of {len(mask)} tiles")
            self.unique_idx = np.cumsum(mask) - 1
            self.paths = self.paths.loc[mask]
        else:
            rands = np.random.RandomState(0).rand(len(self.paths))
            if fold == "train":
                mask = (rands < 0.75)
            elif fold == "val":
                mask = (rands > 0.75) & (rands < 0.90)
            elif fold == "test":
                mask = (rands > 0.90)
            print(f"fold {fold} specified. random splitting. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        if classes is not None:
            mask = self.paths.maxclass.isin(classes)
            print(f"classes {classes} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        if seasons is not None:
            mask = self.paths.season.isin(seasons)
            print(f"seasons {seasons} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]

        # shuffle the tiles once
        self.paths = self.paths.sample(frac=1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths.iloc[index]

        with h5py.File(self.h5file_path, 'r') as data:
            s2 = data[path.h5path + "/s2"][()]
            s1 = data[path.h5path + "/s1"][()]
            label = data[path.h5path + "/lc"][()]

        image, target = data_transform(s1, s2, label)

        image = self.transform(torch.from_numpy(image))
        #reg =  self.regions[path.h5path.split('/')[1]][0]
        #if self.transform_coord is not None:
        #    reg = self.transform_coord(reg)

        t2,c = np.unique(target.flatten(), return_counts=True)
        return image, t2[np.argmax(c)]
