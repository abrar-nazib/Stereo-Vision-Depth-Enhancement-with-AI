"""Load official OpenStereo LightStereo without importing unrelated trainers."""
import sys
import types
from unittest.mock import patch
import torch
import timm
from model import ROOT


def build_lightstereo():
    source=ROOT/'external_models/OpenStereo'
    for name in ('stereo','stereo.modeling','stereo.modeling.models'):
        module=types.ModuleType(name)
        module.__path__=[str(source/name.replace('.','/'))]
        sys.modules[name]=module
    from stereo.modeling.models.lightstereo.lightstereo import LightStereo
    class Config(dict):
        __getattr__=dict.__getitem__
    original=timm.create_model
    def create(*args,**kwargs):
        kwargs['pretrained']=False  # Full stereo checkpoint supplies all parameters.
        net=original(*args,**kwargs)
        if not hasattr(net,'act1'):
            # Current timm integrates ReLU6 into BatchNormAct2d.
            net.act1=torch.nn.Identity()
        return net
    with patch.object(timm,'create_model',create):
        net=LightStereo(Config(MAX_DISP=192,LEFT_ATT=True,AGGREGATION_BLOCKS=[1,2,4],EXPANSE_RATIO=4))
    return net
