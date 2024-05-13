from geopy.geocoders import Nominatim


###################################################
# Functions we want to articulate (APIs calls): ###
###################################################

########################################################################################
# Functions called  in the articulated functions (not directly called by the model): ###
########################################################################################

geolocator = Nominatim(user_agent="MyApp")

def find_precise_place(lat, lon):
    location = geolocator.reverse(str(lat) +", " + str(lon))
    return location.raw.get('display_name', {})

def find_coordinates(address):
    coord = geolocator.geocode(address)
    lat = coord.latitude
    lon = coord.longitude
    return(lat,lon)


def check_city_coordinates(lat = "", lon = "", city = "", **kwargs):
    """
    :param lat: latitude
    :param lon: longitude
    :param city: name of the city

    Checks if the coordinates correspond to the city, if not update the coordinate to correspond to the city
    """
    if lat != "0" and lon != "0":
        reverse = partial(geolocator.reverse, language="en")
        location = reverse(f"{lat}, {lon}")
        address = location.raw.get('address', {})
        city = address.get('city') or address.get('town') or address.get('village') or address.get('county')
    else : 
        reverse = partial(geolocator.reverse, language="en")
        location = reverse(f"{lat}, {lon}")
        address = location.raw.get('address', {})
        city_name = address.get('city') or address.get('town') or address.get('village') or address.get('county')
        if city_name is None :
            city_name = 'not_found'
        print(city_name)
        if city_name.lower() != city.lower():
            coord = geolocator.geocode(city )
            if coord is None:
                coord = geolocator.geocode(city)
            lat = coord.latitude
            lon = coord.longitude
    return lat, lon, city


