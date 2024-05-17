from typing import List, Tuple, Optional, Union


def plot_route(points, vehicle: Union[tuple[float, float], None] = None):
    import plotly.express as px
    
    lats = []
    lons = []

    for point in points:
        lats.append(point["latitude"])
        lons.append(point["longitude"])
    # fig = px.line_geo(lat=lats, lon=lons)
    # fig.update_geos(fitbounds="locations")

    fig = px.line_mapbox(
        lat=lats, lon=lons, zoom=12, height=600, color_discrete_sequence=["red"]
    )

    if vehicle:
        fig.add_trace(
            px.scatter_mapbox(
                lat=[vehicle[0]],
                lon=[vehicle[1]],
                color_discrete_sequence=["blue"],
            ).data[0]
        )

    fig.update_layout(
        mapbox_style="open-street-map",
        # mapbox_zoom=12,
    )
    fig.update_geos(fitbounds="locations")
    fig.update_layout(margin={"r": 20, "t": 20, "l": 20, "b": 20})
    return fig
