"""
Road network cleaning and segmentation.

This script prepares two cleaned road network versions from OpenStreetMap:
- a simple network used for point location estimation in `06_geo-locate.py`
- a 50 m segmented network used in `07_map-matching.py` and `13_aggregate.ipynb`.

Expected Structure:
- Input: geo-files/road_network.shp
- Output: geo-files-processed/road_network_unsplit.graphml
- Output: geo-files-processed/road_network_50m_edges.shp (and nodes)

Prepation: Please prepare your own OSM road network as shapefile before running this script.

"""

import os
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge


# 1. Load and prepare the raw road network from OSM
# We start from a preprocessed road shapefile (exported from OSM),
# build a graph, and standardize its geometry and attributes.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SHP_PATH = PROJECT_ROOT / "geo-files" / "road_network.shp" # Please prepare your own OSM road network as shapefile

OUT_DIR = PROJECT_ROOT / "geo-files-processed"
OUT_UNSPLIT = OUT_DIR / "road_network_unsplit.graphml"
OUT_SPLIT50_GRAPHML = OUT_DIR / "road_network_50m.graphml"  # Optional: remove if you do not need split GraphML

# 50 m split outputs
OUT_SPLIT50_NODES_SHP = OUT_DIR / "road_network_50m_nodes.shp"
OUT_SPLIT50_EDGES_SHP = OUT_DIR / "road_network_50m_edges.shp"

SEGMENT_LENGTH_M = 50
TARGET_CRS_M = "EPSG:3857"
TARGET_CRS_LATLON = "EPSG:4326"

KEEP_COLS = ["cat", "geometry", "highway", "lanes", "maxspeed", "name", "oneway", "ref", "reversed"] # Adapt based on available columns from your OSM shapefile



def _to_linestring(geom):
    """Convert a geometry to a valid LineString, or return None if not valid."""
    if geom is None:
        return None
    if isinstance(geom, LineString):
        return geom if geom.length > 0 else None
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged if merged.length > 0 else None
        if isinstance(merged, MultiLineString):
            parts = [g for g in merged.geoms if isinstance(g, LineString) and g.length > 0]
            if not parts:
                return None
            parts.sort(key=lambda g: g.length, reverse=True)
            return parts[0]
    return None


def _parse_bool(x):
    """Convert various string/number encodings (for example '1.0'/'0.0') into True/False/None."""
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "t", "yes", "y"}:
        return True
    if s in {"false", "f", "no", "n"}:
        return False
    try:
        f = float(s)
        if f == 1.0:
            return True
        if f == 0.0:
            return False
    except Exception:
        pass
    return None


def add_edge_fields_before_export(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Before exporting, add helper fields on each edge:
      - from_node, to_node: node ids
      - link_id: sequential edge id
      - dir: direction flag (kept as 0 here, just a placeholder)

    We also normalize the 'oneway' and 'reversed' attributes in our example to proper booleans
    so that GraphML I/O is more robust.
    """
    edges_sorted = sorted(G.edges(keys=True))
    for link_id, (u, v, k) in enumerate(edges_sorted):
        data = G.edges[u, v, k]
        data["from_node"] = u
        data["to_node"] = v
        data["link_id"] = int(link_id)
        data["dir"] = 0

        if "oneway" in data:
            data["oneway"] = _parse_bool(data["oneway"])
        if "reversed" in data:
            data["reversed"] = _parse_bool(data["reversed"])
    return G


def build_graph_from_edges_shp(shp_path: str) -> nx.MultiDiGraph:
    """Build a MultiDiGraph from a road shapefile."""
    edges_raw = gpd.read_file(shp_path)

    keep = [c for c in KEEP_COLS if c in edges_raw.columns]
    if "geometry" not in keep:
        keep.append("geometry")
    edges_raw = edges_raw[keep].copy()

    edges_raw["geometry"] = edges_raw["geometry"].apply(_to_linestring)
    edges_raw = edges_raw[edges_raw["geometry"].notna()].copy()
    edges_raw = edges_raw[edges_raw["geometry"].apply(lambda g: isinstance(g, LineString) and g.length > 0)].copy()

    if edges_raw.empty:
        raise ValueError("No valid LineString geometries found in the shapefile.")

    # Build node set from line endpoints
    edges_raw["u_coords"] = edges_raw["geometry"].apply(lambda ln: tuple(ln.coords[0]))
    edges_raw["v_coords"] = edges_raw["geometry"].apply(lambda ln: tuple(ln.coords[-1]))

    unique_coords = list(set(edges_raw["u_coords"].tolist() + edges_raw["v_coords"].tolist()))
    unique_coords.sort()

    nodes_data = []
    for osmid, coords in enumerate(unique_coords):
        x, y = coords
        nodes_data.append({"osmid": osmid, "x": float(x), "y": float(y), "geometry": Point(coords)})

    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs=edges_raw.crs).set_index("osmid")
    nodes_gdf["node_id"] = nodes_gdf.index.astype(int)
    coord_to_id = {(row["x"], row["y"]): node_id for node_id, row in nodes_gdf.iterrows()}

    edges_raw["u"] = edges_raw["u_coords"].apply(lambda c: coord_to_id[(float(c[0]), float(c[1]))])
    edges_raw["v"] = edges_raw["v_coords"].apply(lambda c: coord_to_id[(float(c[0]), float(c[1]))])
    edges_raw["key"] = 0
    edges_raw.drop(columns=["u_coords", "v_coords"], inplace=True)

    edges_gdf = edges_raw.drop_duplicates(subset=["u", "v"], keep="first").copy()
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    G = ox.graph_from_gdfs(nodes_gdf, edges_gdf)
    G.graph["crs"] = edges_raw.crs
    return G


def simplify_and_project(G: nx.MultiDiGraph, to_crs_m: str = TARGET_CRS_M) -> nx.MultiDiGraph:
    """Simplify the network topology and project it to a metric CRS."""
    G_s = ox.simplify_graph(G)
    G_m = ox.project_graph(G_s, to_crs=to_crs_m)
    return G_m


def split_graph_edges(G_in: nx.MultiDiGraph, segment_length: float = 50) -> nx.MultiDiGraph:
    """
    Split each edge of the graph into segments of roughly `segment_length` meters.

    This produces a denser network where each segment is approximately the same length,
    which is convenient for later aggregation at fixed spatial resolution (e.g. 50 m in this case).
    """
    G_out = nx.MultiDiGraph()
    G_out.graph = G_in.graph.copy()

    # Copy all nodes
    for n, data in G_in.nodes(data=True):
        G_out.add_node(n, **data)

    int_nodes = [n for n in G_in.nodes() if isinstance(n, (int, np.integer))]
    next_node_id = int(max(int_nodes) + 1) if int_nodes else 0

    for u, v, k, data in list(G_in.edges(keys=True, data=True)):
        geom = data.get("geometry", None)

        if geom is None or not isinstance(geom, LineString) or geom.length == 0:
            G_out.add_edge(u, v, key=k, **data)
            continue

        total_len = float(geom.length)

        if total_len <= segment_length:
            d2 = data.copy()
            d2["length"] = total_len
            G_out.add_edge(u, v, key=k, **d2)
            continue

        num_segments = int(total_len // segment_length)
        center_offset = (total_len - num_segments * segment_length) / 2.0

        split_points = [geom.interpolate(center_offset + i * segment_length) for i in range(1, num_segments)]

        nodes_seq = [u]
        coords_seq = [tuple(geom.coords[0])]

        for pt in split_points:
            new_id = next_node_id
            next_node_id += 1
            G_out.add_node(new_id, x=float(pt.x), y=float(pt.y), geometry=pt)
            nodes_seq.append(new_id)
            coords_seq.append((float(pt.x), float(pt.y)))

        nodes_seq.append(v)
        coords_seq.append(tuple(geom.coords[-1]))

        for i in range(len(nodes_seq) - 1):
            seg_geom = LineString([coords_seq[i], coords_seq[i + 1]])
            seg_data = data.copy()
            seg_data["geometry"] = seg_geom
            seg_data["length"] = float(seg_geom.length)
            G_out.add_edge(nodes_seq[i], nodes_seq[i + 1], key=0, **seg_data)

    return G_out


def export_graphml_latlon(G_m: nx.MultiDiGraph, out_path: Path) -> None:
    """Project graph back to WGS84 and save to GraphML."""
    G_ll = ox.project_graph(G_m, to_crs=TARGET_CRS_LATLON)
    ox.save_graphml(G_ll, out_path)


def export_split50_shp_latlon(G_split_m: nx.MultiDiGraph, nodes_shp: Path, edges_shp: Path) -> None:
    """Project the split network back to WGS84 and save nodes/edges as shapefiles."""
    G_ll = ox.project_graph(G_split_m, to_crs=TARGET_CRS_LATLON)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G_ll)

    gdf_nodes["node_id"] = gdf_nodes.index.astype(int)

    os.makedirs(os.path.dirname(nodes_shp), exist_ok=True)
    os.makedirs(os.path.dirname(edges_shp), exist_ok=True)

    gdf_nodes.to_file(nodes_shp)
    gdf_edges.to_file(edges_shp)



def main() -> None:
    """Clean, simplify, split, and export the road network."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SHP_PATH.exists():
        print(f"Error: Raw shapefile not found at {SHP_PATH}")
        return

    print(f"Reading SHP from: {SHP_PATH.relative_to(PROJECT_ROOT)}")
    G0 = build_graph_from_edges_shp(str(SHP_PATH))

    print("Simplifying + projecting to metric system...")
    G_m = simplify_and_project(G0, to_crs_m=TARGET_CRS_M)

    print("Exporting UNSPLIT graphml (for 06_geo-locate.py)...")
    G_m = add_edge_fields_before_export(G_m)
    export_graphml_latlon(G_m, OUT_UNSPLIT)

    print(f"Splitting edges into {SEGMENT_LENGTH_M}m segments...")
    G_split_m = split_graph_edges(G_m, segment_length=SEGMENT_LENGTH_M)

    print("Exporting SPLIT 50m data (for 07_map-matching.py)...")
    G_split_m = add_edge_fields_before_export(G_split_m)
    
    # Export GraphML
    export_graphml_latlon(G_split_m, OUT_SPLIT50_GRAPHML)
    
    # Export Shapefiles
    export_split50_shp_latlon(G_split_m, OUT_SPLIT50_NODES_SHP, OUT_SPLIT50_EDGES_SHP)

    print("\nDone. Preprocessing outputs saved to:")
    print(f" {OUT_UNSPLIT.relative_to(PROJECT_ROOT)}")
    print(f" {OUT_SPLIT50_EDGES_SHP.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()


