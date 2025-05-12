import os

import numpy as np
from mmsdk import mmdatasdk as md

DATASET = md.cmu_mosei
DATA_PATH = 'cmumosi/'


def avg(intervals: np.array, features: np.array) -> np.array:
    # 虽然没有用到intervals，但是还是要作为参数，不然会报错
    try:
        return np.average(features, axis=0)
    except:
        return features

if __name__ == "__main__":
    try:
        md.mmdataset(DATASET.highlevel, DATA_PATH)
    except:
        print('have been downloaded')
    visual_field = 'CMU_MOSEI_VisualFacet42.csd'
    acoustic_field = 'CMU_MOSEI_COVAREP.csd'
    text_field = 'CMU_MOSEI_TimestampedWordVectors.csd'

    features = [
        text_field,
        visual_field,
        acoustic_field
    ]
    recipe = {feat: os.path.join(DATA_PATH, feat) for feat in features}
    dataset = md.mmdataset(recipe)
    print(list(dataset.keys()))  # 数据集的key
    print("=" * 50)

    print(list(dataset[visual_field].keys())[:10])  # 视觉模态的前十个key，这里即前十个id
    print("=" * 50)

    # 第十五个视频的keys，即那个一个元组
    some_id = list(dataset[visual_field].keys())[15]
    print(list(dataset[visual_field][some_id].keys()))
    print("=" * 50)

    # 看一下时间戳的shape
    print(list(dataset[visual_field][some_id]['intervals'].shape))
    # print(list(dataset[visual_field][some_id]['intervals']))
    print("=" * 50)

    # 看一下每一个模态的shape
    print(list(dataset[visual_field][some_id]['features'].shape))
    print(list(dataset[text_field][some_id]['features'].shape))
    print(list(dataset[acoustic_field][some_id]['features'].shape))
    # 不同的模态有不同的time step
    print("Different modalities have different number of time steps!")

    dataset.align(text_field, collapse_functions=[avg])