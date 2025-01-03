import json

import pandas as pd


if __name__ == '__main__':
    # Parameters
    query_path = "../query.csv"
    cluster_result_path = "./cluster_result.json"
    query_result_path = "./submission.csv"

    df = pd.read_csv(query_path)
    with open(cluster_result_path, 'r') as f:
        cluster_result = json.load(f)

    def generate_label(fileA, fileB):
        """
        Judge whether two images are in the same cluster.
        """
        if cluster_result[str(fileA)+".png"] == cluster_result[str(fileB)+".png"]:
            return 1
        else:
            return 0

    df['label'] = df.apply(lambda row: generate_label(row['fileA'], row['fileB']), axis=1)
    df.to_csv(query_result_path, index=False)
