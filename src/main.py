#!/usr/bin/env python3

import sys
import re
import csv
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

sys.path.append(r'..')

from lib.dataset import Dataset
from lib.chatGPT import ChatGPT


def main():
    dataset_path = "../data/Manipulated-2016-2020.xlsx"
    result_file_path = "../results/feature importance.csv"
    gpt = ChatGPT()

    try:
        # dataset = Dataset(path=dataset_path, label='label').drop(column='id').drop(column='g')
        dataset = Dataset(path=dataset_path, label='Label')
        for i in range(1, 21):
            dataset.x[f'F{i}'] = dataset.x[f'F{i}'].astype('category')
        cat_columns = dataset.x.select_dtypes(['category']).columns
        dataset.x[cat_columns] = dataset.x[cat_columns].apply(lambda x: x.cat.codes)

        serialized_data = np.array(list(zip(Dataset.serialize(dataset.x), dataset.y)))
        np.random.shuffle(serialized_data)
        batch_count = 100

        gpt.add_prompt({
            "role": "system",
            "content": "as a data scientist, "
                       "given the data as "
                       f"{', '.join(['feature:value' for _ in range(len(dataset.x.columns))])} -> label, "
                       "rank all the features based on their importance "
                       "(specify the feature importance weight for each feature).\n"
                       "when you get all the data, just list important features.\n"
                       "for example list the most important features exactly like the following in one line:\n"
                       f"{', '.join([f'feature {i + 1}' for i in range(len(dataset.x.columns))])}\n"
        })
        print(f'DEBUG: system prompt -> {gpt.prompts[0]["content"]}')

        result_fields = ['method', 'values']
        csv_file = open(result_file_path, 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(result_fields)

        def extract_features(string: str) -> [str]:
            _match = re.findall(r'\w+', string)
            return np.array(_match) if _match is not None else np.array([])

        gpt_response = gpt.add_prompt({
            "role": "user",
            "content": "\n".join([f"{s} -> {l}" for s, l in serialized_data[:batch_count]])
        }).ask()

        gpt_extracted_features = extract_features(gpt_response)

        csv_writer.writerow(["ChatGPT", ", ".join(gpt_extracted_features)])

        def information_gain(x, y):
            arr = pd.Series(mutual_info_classif(x, y, random_state=4), index=x.columns).sort_values().keys().tolist()
            arr.reverse()
            return np.array(arr)

        def correlation(x, y):
            _y = y.copy(deep=True)
            _y[_y == 'Yes'] = 1
            _y[_y == 'No'] = 0
            return np.array(pd.DataFrame({'Feature': x.columns, 'Correlation': x.corrwith(_y).abs()})
                            .sort_values(by='Correlation', ascending=False)['Feature'].tolist())

        def pca(x, y):
            return np.array(pd.DataFrame({
                'Feature': x.columns,
                'Loading': np.abs(PCA(n_components=1).fit(x, y).components_.T)[:, 0],
            }).sort_values(by='Loading', ascending=False)['Feature'].tolist())

        feature_selection_methods = {
            'IG': information_gain,
            'Corr': correlation,
            'PCA': pca,
        }

        feature_selection_intersections = {
            'IG ∩ Corr': [],
            'IG ∩ PCA': [],
            'Corr ∩ PCA': [],
        }

        feature_selection_union = {
            'IG ∪ Corr': [],
            'IG ∪ PCA': [],
            'Corr ∪ PCA': [],
        }

        feature_importance = [(name, np.array(func(dataset.x, dataset.y))) for name, func in
                              feature_selection_methods.items()]

        top_importance_count = 10

        def intersection(lst1, lst2):
            lst3 = [value for value in lst1 if value in lst2]
            return lst3

        for method, features in feature_importance:
            csv_writer.writerow([method, ", ".join(features)])
            for _name, _values in feature_importance:
                intersection_name = f'{method} ∩ {_name}'
                union_name = f'{method} ∪ {_name}'

                if intersection_name in feature_selection_intersections:
                    feature_selection_intersections[intersection_name] = intersection(features[:top_importance_count],
                                                                                      _values[:top_importance_count])
                if union_name in feature_selection_union:
                    feature_selection_union[union_name] = np.union1d(features[:top_importance_count],
                                                                     _values[:top_importance_count])[::-1]

        for method, features in feature_selection_intersections.items():
            csv_writer.writerow([method, ", ".join(features)])

        for method, features in feature_selection_union.items():
            csv_writer.writerow([method, ", ".join(features)])

        features_importance_weight = {
            'IG': {feature: 0 for feature in dataset.x.columns},
            'Corr': {feature: 0 for feature in dataset.x.columns},
            'PCA': {feature: 0 for feature in dataset.x.columns},
        }

        features_importance_total_score = {}

        for method, features in feature_importance:
            for i, feature in zip(range(len(features), 0, -1), features):
                features_importance_weight[method][feature] += i

        for method, weights in features_importance_weight.items():
            for f, w in weights.items():
                if f not in features_importance_total_score:
                    features_importance_total_score[f] = w
                else:
                    features_importance_total_score[f] += w

        # features_importance_feature_score = {
        #     'method': {
        #         'feature': score
        #     }
        # }
        features_importance_feature_score = {}

        for method, weights in features_importance_weight.items():
            total_weights = sum(list(weights.values()))
            features_importance_feature_score[method] = {}
            for feature, weight in weights.items():
                features_importance_feature_score[method][feature] = (weight / total_weights) * 100

        plt.figure(figsize=(20, 10))
        x = list(dataset.x.columns)
        x_axis = np.arange(len(x))

        methods = list(feature_selection_methods.keys())
        width = .2
        rs = np.arange(-width, +width + .1, width)
        for r, (i, method) in zip(rs, enumerate(methods)):
            plt.bar(x_axis + r, list(features_importance_feature_score[method].values()), width, label=method)

        plt.xticks(x_axis, x)
        plt.xlabel("Features")
        plt.ylabel("Score")
        plt.title("Feature Importance")
        plt.legend()
        plt.show()

    except FileNotFoundError:
        print(f'dataset at {dataset_path} not found')
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    main()

