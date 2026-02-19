from __future__ import annotations


def render_topic_tree(top_clusters: list[dict], sub_clusters: list[dict]) -> str:
    lines = ["Topic Tree", "========="]
    sub_map = {}
    for item in sub_clusters:
        sub_map.setdefault(item["parent_cluster_id"], []).append(item)

    for cluster in sorted(top_clusters, key=lambda x: x["cluster_id"]):
        lines.append(f"- [{cluster['cluster_id']}] {cluster['label']} (n={cluster['size']})")
        for sub in sorted(sub_map.get(cluster["cluster_id"], []), key=lambda x: x["subcluster_id"]):
            lines.append(f"  - [{cluster['cluster_id']}.{sub['subcluster_id']}] {sub['label']} (n={sub['size']})")
    return "\n".join(lines)
