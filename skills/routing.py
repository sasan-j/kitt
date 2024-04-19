def calculate_route():
    api_key = "api_key"
    origin = "49.631997,6.171029"
    destination = "49.586745,6.140002"

    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin}:{destination}/json?key={api_key}"
    response = requests.get(url)
    data = response.json()

    lats = []
    lons = []

    for point in data['routes'][0]['legs'][0]['points']:
        lats.append(point['latitude'])
        lons.append(point['longitude'])
    # fig = px.line_geo(lat=lats, lon=lons)
    # fig.update_geos(fitbounds="locations")

    fig = px.line_mapbox(lat=lats, lon=lons, zoom=12, height=600)

    fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=12, mapbox_center_lat=lats[0], mapbox_center_lon=lons[0])
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig