'''
This script comes from the Stanford project. It gives command line access to functions
to download Street View images, based on the IDs in the metadata file. 
I've minimally edited the download functions just to match the file structure.
'''
import fire

# from plot import plot_all
# from streetview import (download_streetview_image,
                        # calculate_coverage,
                        # calculate_zone,
                        # calculate_road_length)
from streetview import download_streetview_image


if __name__ == "__main__":
    fire.Fire()
