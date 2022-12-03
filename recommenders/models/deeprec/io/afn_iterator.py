
from recommenders.models.deeprec.io.iterator import FFMTextIterator

class AFNFFMTextIterator(FFMTextIterator):
    def __init__(self, batch_size, feature_cnt, field_cnt,  col_spliter=" ", ID_spliter="%"):
        self.batch_size = batch_size
        self.feature_cnt = feature_cnt
        self.field_cnt = field_cnt
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.labels = "label"
        self.fm_feat_indices = "oh_indices"
        self.fm_feat_values = "oh_values"
        self.fm_feat_shape = "oh_shape"

        self.dnn_feat_indices = "feat_indices"
        self.dnn_feat_values = "values"
        self.dnn_feat_weights = "weights"
        self.dnn_feat_shape = "shape"
    # def gen_feed_dict(self, data_dict):
    #     feed_dict = super(AFNFFMTextIterator, self).gen_feed_dict(data_dict)
    #     feed_dict2 = {}
    #     for n, feat in feed_dict.items():
    #         feed_dict2[n] = feat
    #     return feed_dict2