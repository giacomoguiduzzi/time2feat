import os
import pickle
from pickle import UnpicklingError
from typing import Optional

import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from time2feat.utils.importance_old import feature_selection
from time2feat.extraction.extractor import feature_extraction
from time2feat.model.clustering import ClusterWrapper


class Time2Feat(object):

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        batch_size=100,
        p=1,
        model_type="KMeans",
        transform_type="std",
        score_mode="simple",
        strategy="sk_base",
        k_best=False,
        pre_transform=False,
        top_k=None,
        pfa_value=0.9,
    ):
        """
        Initialize time2feat method with specified parameters.

        Parameters:
        - n_clusters (int): Number of clusters to be used. If None, the number of selected features will be used.
        - batch_size (int, optional): Batch size for processing (default is 100).
        - p (int, optional): Some parameter `p` (default is 1).
        - model_type (str, optional): Type of model to use (default is 'kMeans').
        - transform_type (str, optional): Type of data normalization (default is 'std').
        - score_mode (str, optional): Mode for scoring (default is 'simple').
        - strategy (str, optional): Strategy of features selection to be used (default is 'sk_base').
        - k_best (bool, optional): Whether to use k-best selection (default is False).
        - pre_transform (bool, optional): Whether to apply pre-transformation (default is False).
        - top_k (list of int, optional): List of top-k values (default is [1]).
        - pfa_value (float, optional): Some value `pfa_value` (default is 0.9).
        """
        if top_k is None:
            top_k = [1]
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.p = p
        self.model_type = model_type
        self.transform_type = transform_type
        self.score_mode = score_mode
        self.strategy = strategy
        self.k_best = k_best
        self.pre_transform = pre_transform
        self.top_k = top_k
        self.pfa_value = pfa_value
        self.top_feats = list()
        self.top_ext_feats = list()
        self._df_feats = None
        self.y_pred = None
        self.top_feats_variance = None
        self.top_ext_feats_variance = None

    def fit(
        self,
        X,
        labels=None,
        external_feat: pd.DataFrame = None,
        select_external_feat: bool = False,
        save_features_extracted: str = "last_experiment",
    ):
        if labels is None:
            labels = dict()

        path_temp = save_features_extracted
        print("Sto salvando o usando le Features di: " + path_temp)

        if not os.path.exists(path_temp):
            df_feats = feature_extraction(X, batch_size=self.batch_size, p=self.p)
            with open(path_temp, "wb") as f:
                pickle.dump(df_feats, f)
        else:
            try:
                df_feats = pickle.load(open(path_temp, "rb"))
            except UnpicklingError:
                print(
                    "Error in loading the pickle file. It looks like the file is corrupt. Recomputing the features."
                )
                df_feats = feature_extraction(X, batch_size=self.batch_size, p=self.p)
                with open(path_temp, "wb") as f:
                    pickle.dump(df_feats, f)

        context = {
            "model_type": self.model_type,
            "transform_type": self.transform_type,
            "score_mode": self.score_mode,
            "strategy": self.strategy,
            "k_best": self.k_best,
            "pre_transform": self.pre_transform,
            "top_k": self.top_k,
            "pfa_value": self.pfa_value,
        }

        top_features_dict = feature_selection(
            df_feats,
            labels=labels,
            context=context,
            external_feat=external_feat,
            select_external_feat=select_external_feat,
        )

        top_features = top_features_dict["ts_feats"]
        top_features_variance = top_features_dict["ts_feats_var"]
        top_external_features = top_features_dict["ext_feats"]
        top_external_features_variance = top_features_dict["ext_feats_var"]

        if external_feat is not None:
            df_feats = pd.concat(
                [df_feats[top_features], external_feat[top_external_features]], axis=1
            )
            self.top_feats = top_features
            self.top_ext_feats = top_external_features
        else:
            df_feats = df_feats[top_features]
            self.top_feats = top_features
        """if external_feat is not None:
            _df_feats = pd.concat([_df_feats, external_feat], axis=1)"""

        self.top_feats_variance = top_features_variance
        self.top_ext_feats = top_external_features
        self.top_ext_feats_variance = top_external_features_variance

        # return _df_feats

        # scaler = StandardScaler()
        # df_standardized = scaler.fit_transform(_df_feats)
        #
        # # PCA on the selected features
        # pca = PCA()
        # pca.fit(df_standardized)
        #
        # # get the variance for the variables
        # explained_variance = pca.explained_variance_ratio_
        # cumulative_explained_variance = explained_variance.cumsum()

        # select significant features
        # n_components = next(i for i, total_var in enumerate(cumulative_explained_variance) if total_var >= 0.95) + 1

        # Get the principal components
        # components = pca.components_[:n_components]

        # Get the most important features for each component
        # most_important_features = []
        # for component in components:
        #     feature_indices = component.argsort()[-n_components:][::-1]
        #     most_important_features.extend(_df_feats.columns[feature_indices])

        # Remove duplicates while preserving order
        # most_important_features = list(dict.fromkeys(most_important_features))

        # Select the most important features from the DataFrame
        # df_selected_features = _df_feats[most_important_features]

        # print(f"Selected {n_components} features: {most_important_features}")

        # return df_selected_features

        if self.n_clusters is None:
            scaler = StandardScaler()
            df_standardized = scaler.fit_transform(df_feats)

            # PCA on the selected features
            pca = PCA()
            pca.fit(df_standardized)

            # get the variance for the variables
            explained_variance = pca.explained_variance_ratio_

            n_clusters = KneeLocator(
                np.asarray([(range(len(explained_variance)))]).squeeze(),
                np.asarray(explained_variance).squeeze(),
                curve="convex",
                direction="decreasing",
            ).knee

            self.n_clusters = n_clusters

        else:
            n_clusters = self.n_clusters

        self._df_feats = df_feats

        return df_feats, n_clusters

    def predict(self):
        model = ClusterWrapper(
            n_clusters=self.n_clusters,
            model_type=self.model_type,
            transform_type=self.transform_type,
        )
        self.y_pred = model.fit_predict(self._df_feats)
        return self.y_pred

    def fit_predict(
        self,
        X,
        labels=None,
        external_feat: pd.DataFrame = None,
        select_external_feat: bool = False,
        save_features_extracted: str = "last_experiment",
    ):
        if labels is None:
            labels = {}
        df_feats, n_clusters = self.fit(
            X,
            labels,
            external_feat,
            select_external_feat=select_external_feat,
            save_features_extracted=save_features_extracted,
        )
        model = ClusterWrapper(
            n_clusters=n_clusters,
            model_type=self.model_type,
            transform_type=self.transform_type,
        )
        y_pred = model.fit_predict(df_feats)
        return y_pred
