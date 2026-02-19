#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Part 3: topic clustering and tree")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Number of sampled documents")
    parser.add_argument("--test-size", type=float, default=0.2, help="Unused; present for interface consistency")
    parser.add_argument("--vectorizer", choices=["bow", "tfidf"], default="tfidf", help="Unused; kept for uniform CLI")
    parser.add_argument("--st-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for output artifacts")
    return parser


def run(args):
    import numpy as np
    from sklearn.cluster import KMeans

    from src.clustering import elbow_search, nearest_docs_to_centroid, save_elbow
    from src.config import ensure_outputs_dir
    from src.data import load_dataset
    from src.features import encode_texts
    from src.labeling import get_labeler
    from src.reporting import plot_elbow
    from src.topic_tree import render_topic_tree

    out_dir = ensure_outputs_dir(args.outputs_dir)
    data = load_dataset(n_samples=args.n_samples, seed=args.seed)
    embeddings = encode_texts(data.texts, args.st_model)

    elbow = elbow_search(embeddings, ks=range(2, 10), seed=args.seed)
    save_elbow(out_dir / "elbow.json", elbow)
    plot_elbow(elbow["ks"], elbow["inertias"], elbow["chosen_k"], out_dir / "elbow.png")

    km = elbow["model"]
    labels = km.labels_
    labeler = get_labeler()

    top_clusters = []
    for cid in range(elbow["chosen_k"]):
        idx = np.where(labels == cid)[0]
        nearest = nearest_docs_to_centroid(embeddings, idx, km.cluster_centers_[cid], top_n=8)
        snippets = [data.texts[i][:280].replace("\n", " ") for i in nearest]
        lbl = labeler.label(snippets)
        top_clusters.append(
            {
                "cluster_id": cid,
                "size": int(len(idx)),
                "representative_snippets": snippets,
                "label": lbl.get("label", "Unknown"),
                "rationale": lbl.get("rationale", ""),
            }
        )

    (out_dir / "clusters_top_level.json").write_text(json.dumps(top_clusters, indent=2), encoding="utf-8")

    two_largest = sorted(top_clusters, key=lambda x: x["size"], reverse=True)[:2]
    sub_clusters = []
    for cluster in two_largest:
        cid = cluster["cluster_id"]
        idx = np.where(labels == cid)[0]
        sub_km = KMeans(n_clusters=3, random_state=args.seed, n_init=10)
        sub_km.fit(embeddings[idx])
        sub_labels = sub_km.labels_
        for sub_id in range(3):
            sub_idx_local = np.where(sub_labels == sub_id)[0]
            actual_idx = idx[sub_idx_local]
            nearest = nearest_docs_to_centroid(embeddings, actual_idx, sub_km.cluster_centers_[sub_id], top_n=8)
            snippets = [data.texts[i][:280].replace("\n", " ") for i in nearest]
            lbl = labeler.label(snippets)
            sub_clusters.append(
                {
                    "parent_cluster_id": int(cid),
                    "subcluster_id": int(sub_id),
                    "size": int(len(actual_idx)),
                    "representative_snippets": snippets,
                    "label": lbl.get("label", "Unknown"),
                    "rationale": lbl.get("rationale", ""),
                }
            )

    (out_dir / "clusters_sub_level.json").write_text(json.dumps(sub_clusters, indent=2), encoding="utf-8")

    tree_text = render_topic_tree(top_clusters, sub_clusters)
    (out_dir / "topic_tree.txt").write_text(tree_text, encoding="utf-8")
    print(tree_text)
    return {"chosen_k": elbow["chosen_k"], "top_clusters": top_clusters, "sub_clusters": sub_clusters, "tree_text": tree_text}


if __name__ == "__main__":
    parser = get_parser()
    run(parser.parse_args())
