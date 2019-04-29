"""
Python Class to build multiple layers of PredNet programmatically.
My original solution was manually wired, so this should be a bit cleaner to understand.

This class will build our PredNet in a bottom-up sweep, where error is propagated to higher layers as higher dimensional
representations. The goal is to have the higher layers learn to predict error, and thus limit error on the 0-th layer.

Written By: Zachary Kirby (zjk2775@louisiana.edu)
"""
import PredNetModel
import tensorflow as tf


class MultiLayerPredNet:

    def __init__(self, **kwargs):
        self.layer_count = kwargs.pop("layer_count", 1)
        self.img_width = kwargs.pop('img_width', 64)
        self.img_height = kwargs.pop('img_height', 64)
        self.num_unrollings = kwargs.pop('num_unrollings', 3)
        self.channels = kwargs.pop('channels', 1)

    # Build a python list of PredNet models that can link together
    #       Will need some extra lifting to work (for now!)
    def build_layers_(self):

        # Init vars that will change with higher layers
        new_channels = int(self.channels)
        new_img_height = int(self.img_height)
        new_img_width = int(self.img_width)

        # Create the layers, append to list
        #       This could be altered to a dictionary, maybe later?
        prednet_models = list()
        for layer in range(self.layer_count):

            # Build our custom PredNetModel object
            #print('LAYER: %d\nCH: %d\nHGT: %d\nWID: %d\n' % (layer, new_channels, new_img_height, new_img_height))
            prednet_layer = PredNetModel.PredNetModel(NUM_UNROLLINGS=self.num_unrollings,
                                                      IN_CHANNELS=new_channels,
                                                      CORE_CHANNELS=new_channels,
                                                      IM_SZ_HGT=new_img_height,
                                                      IM_SZ_WID=new_img_width,
                                                      LAYER=layer)

            # This is a bit strange, BUT the second layer uses a REALLY up-sampled version of the error
            # We use Conv2D to ensure our pixel values maintain spatial information when passed higher
            if layer == 0:
                # This code DOES NOT alter the output, it just tells the layer what to expect
                # This takes care of tensorflow shape mis-matching and placeholders
                new_channels = int(32)  # hard coded until I think about it more
                new_img_height = int(new_img_height / 2)
                new_img_width = int(new_img_width / 2)

            # Squish the error into a different shape (layers =  {0,..,n} \ {1})
            # Basically, we reduce the img size by half (using strides of 2) and double the depth.
            #     This adds a bit more high dimensional information to higher layers
            if layer != 0:
                # Slight up-sample, since the image will actually have its total size halved
                new_channels = int(new_channels * 2)
                new_img_height = int(new_img_height / 2)
                new_img_width = int(new_img_width / 2)

            # Add the layer, it's ready to deploy!
            prednet_models.append(prednet_layer)

        # Finally, return the list of pre-configured PredNet model layers
        # All they need is a feed dict!
        return prednet_models
