import osmnx as ox
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from tqdm import tqdm
from typing import List, Tuple, Union, Optional


class TimebandGenerator:
    def __init__(self,
                 place: Union[str, Tuple[float, float]],
                 network_type: str = "walk",
                 trip_times: List[int] = [5, 10, 15, 20, 25],
                 travel_speed: float = 4.5,
                 dist: Optional[float] = None,
                 dist_type: str = "network",
                 crs: str = "EPSG:4326"):
        """
        Initialize the TimebandGenerator with parameters.
        """
        self.place = place
        self.network_type = network_type
        self.trip_times = trip_times
        self.travel_speed = travel_speed
        self.dist = dist
        self.dist_type = dist_type
        self.crs = crs

        if self.dist is None:
            max_trip_time = max(self.trip_times)  # in minutes
            self.dist = (self.travel_speed * 1000 / 60) * max_trip_time  # in meters

        self.G, self.lat, self.lon = self._load_graph()
        self.G_projected = self._project_graph()
        self.center_node = self._get_center_node()
        self._add_edge_travel_time()

    def _load_graph(self) -> Tuple[nx.MultiDiGraph, float, float]:
        """
        Load the graph from OSM and get latitude and longitude of the place.
        """
        if isinstance(self.place, tuple):
            G = ox.graph_from_point(self.place, network_type=self.network_type, dist=self.dist,
                                    dist_type=self.dist_type)
            lat, lon = self.place
        else:
            G = ox.graph_from_address(self.place, network_type=self.network_type, dist=self.dist,
                                      dist_type=self.dist_type)
            geocode_result = ox.geocode(self.place)
            lat, lon = geocode_result[0], geocode_result[1]
        return G, lat, lon

    def _project_graph(self) -> nx.MultiDiGraph:
        """
        Project the graph to UTM.
        """
        return ox.project_graph(self.G)

    def _get_center_node(self) -> int:
        """
        Find the nearest node to the provided coordinates.
        """
        return ox.distance.nearest_nodes(self.G, X=self.lon, Y=self.lat)

    def _add_edge_travel_time(self) -> None:
        """
        Add an edge attribute for time in minutes required to traverse each edge.
        """
        meters_per_minute = self.travel_speed * 1000 / 60  # km per hour to m per minute
        for u, v, k, data in self.G_projected.edges(data=True, keys=True):
            data["time"] = data["length"] / meters_per_minute

    def generate_timeband_gdf(self) -> gpd.GeoDataFrame:
        """
        Generate a GeoDataFrame of nodes within specified trip times from the given place.
        """
        gdf_nodes = ox.graph_to_gdfs(self.G, edges=False)

        columns = ["node_id", "timeband", "geometry"]
        timeband_gdf = gpd.GeoDataFrame(columns=columns)
        timeband_gdf.crs = self.crs

        node_data = [
            {"node_id": node, "timeband": trip_time,
             "geometry": Point(gdf_nodes.loc[node].geometry.x, gdf_nodes.loc[node].geometry.y)}
            for trip_time in tqdm(self.trip_times, desc="Generating timebands")
            for node in nx.ego_graph(self.G_projected, self.center_node, radius=trip_time, distance="time").nodes
        ]

        timeband_gdf = gpd.GeoDataFrame(node_data, columns=columns).set_crs(self.crs)
        return timeband_gdf

    @staticmethod
    def calculate_minimum_bounding_geometry(timeband_gdf: gpd.GeoDataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Calculate the minimum bounding geometry for each timeband.
        """
        bounding_geometries = []

        grouped = timeband_gdf.groupby('timeband')

        for timeband, group in grouped:
            bounding_geometry = group.unary_union.convex_hull
            bounding_geometries.append({'timeband': timeband, 'geometry': bounding_geometry})

        bounding_gdf = gpd.GeoDataFrame(bounding_geometries, columns=['timeband', 'geometry'])
        bounding_gdf.crs = crs

        return bounding_gdf

    @staticmethod
    def subtract_overlapping_polygons(bounding_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Subtract overlapping parts of polygons from smaller timebands from larger timebands.
        """
        bounding_gdf = bounding_gdf.sort_values(by='timeband').reset_index(drop=True)

        adjusted_geometries = []
        previous_geometry = None

        for i, row in bounding_gdf.iterrows():
            current_geometry = row['geometry']
            if previous_geometry is not None:
                current_geometry = current_geometry.difference(previous_geometry)
            adjusted_geometries.append({'timeband': row['timeband'], 'geometry': current_geometry})
            previous_geometry = row['geometry']

        adjusted_gdf = gpd.GeoDataFrame(adjusted_geometries, columns=['timeband', 'geometry'])
        adjusted_gdf.crs = bounding_gdf.crs

        return adjusted_gdf

