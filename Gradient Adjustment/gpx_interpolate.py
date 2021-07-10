# Copyright (c) 2019 Remi Salmon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# imports
import gpxpy

import numpy as np

from datetime import datetime
from scipy.interpolate import interp1d, splprep, splev

# constants
EARTH_RADIUS = 6371e3 # meters

# functions
def gpx_interpolate(gpx_data, res, deg = 1):
    # input: gpx_data = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}
    #        res = float
    #        deg = int
    # output: gpx_data_interp = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}
    
    if not type(deg) is int:
        raise TypeError('deg must be int')

    if not 1 <= deg <= 5:
        raise ValueError('deg must be in [1-5]')

    if not len(gpx_data['lat']) > deg:
        raise ValueError('number of data points must be > deg')
        
    print('Interpolating ...')

    # interpolate spatial data
    _gpx_data = gpx_remove_duplicate(gpx_data)
    
#     print(2)

    _gpx_dist = gpx_calculate_distance(_gpx_data)
    
#     print(3)

    x = [_gpx_data[i] for i in ('lat', 'lon', 'ele') if _gpx_data[i]]
    
#     print(4)

    tck, _ = splprep(x, u = np.cumsum(_gpx_dist), k = deg, s = 0)
    
#     print(5)

    u_interp = np.linspace(0, np.sum(_gpx_dist), num = 1+int(np.sum(_gpx_dist)/res))
    x_interp = splev(u_interp, tck)
    
#     print(6)

    # interpolate time data linearly to preserve monotonicity
    if _gpx_data['tstamp']:
        f = interp1d(np.cumsum(_gpx_dist), _gpx_data['tstamp'], fill_value = 'extrapolate')

        tstamp_interp = f(u_interp)
        
#     print(7)

    gpx_data_interp = {'lat':list(x_interp[0]),
                       'lon':list(x_interp[1]),
                       'ele':list(x_interp[2]) if gpx_data['ele'] else None,
                       'tstamp':list(tstamp_interp) if gpx_data['tstamp'] else None,
                       'tzinfo':gpx_data['tzinfo']}
    
#     print(8)

    return gpx_data_interp

def gpx_calculate_distance(gpx_data, use_ele = True):
    # input: gpx_data = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}
    #        use_ele = bool
    # output: gpx_dist = numpy.ndarray[float]

    gpx_dist = np.zeros(len(gpx_data['lat']))

    for i in range(len(gpx_dist)-1):
        lat1 = np.radians(gpx_data['lat'][i])
        lon1 = np.radians(gpx_data['lon'][i])
        lat2 = np.radians(gpx_data['lat'][i+1])
        lon2 = np.radians(gpx_data['lon'][i+1])

        delta_lat = lat2-lat1
        delta_lon = lon2-lon1

        c = 2.0*np.arcsin(np.sqrt(np.sin(delta_lat/2.0)**2+np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2.0)**2)) # haversine formula

        dist_latlon = EARTH_RADIUS*c # great-circle distance

        if gpx_data['ele'] and use_ele:
            dist_ele = gpx_data['ele'][i+1]-gpx_data['ele'][i]

            gpx_dist[i+1] = np.sqrt(dist_latlon**2+dist_ele**2)
        else:
            gpx_dist[i+1] = dist_latlon

    return gpx_dist

def gpx_calculate_speed(gpx_data):
    # input: gpx_data = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}
    # output: gpx_speed = numpy.ndarray[float]

    gpx_dist = gpx_calculate_distance(gpx_data)

    gpx_speed = gpx_dist/np.concatenate(([1.0], np.diff(gpx_data['tstamp'])))

    gpx_speed = np.nan_to_num(gpx_speed, nan = 0.0)

    return gpx_speed

def gpx_remove_duplicate(gpx_data):
    # input: gpx_data = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}
    # output: gpx_data_nodup = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}

    gpx_dist = gpx_calculate_distance(gpx_data)

    i_dist = np.concatenate(([0], np.nonzero(gpx_dist)[0])) # keep gpx_dist[0] = 0

    if not len(gpx_dist) == len(i_dist):
        print('Removed {} duplicate trackpoint(s)'.format(len(gpx_dist)-len(i_dist)))

    gpx_data_nodup = {'lat':[], 'lon':[], 'ele':[], 'tstamp':[], 'tzinfo':gpx_data['tzinfo']}

    for k in ('lat', 'lon', 'ele', 'tstamp'):
        gpx_data_nodup[k] = [gpx_data[k][i] for i in i_dist] if gpx_data[k] else None

    return gpx_data_nodup

def gpx_read(gpx_file):
    # input: gpx_file = str
    # output: gpx_data = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}

    gpx_data = {'lat':[], 'lon':[], 'ele':[], 'tstamp':[], 'tzinfo':None}

    i = 0
    i_latlon = []
    i_tstamp = []

    with open(gpx_file, 'r') as file:
        gpx = gpxpy.parse(file)

        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    gpx_data['lat'].append(point.latitude)
                    gpx_data['lon'].append(point.longitude)

                    i_latlon.append(i)

                    try:
                        gpx_data['ele'].append(point.elevation)
                    except:
                        pass

                    try:
                        gpx_data['tstamp'].append(point.time.timestamp())
                    except:
                        pass
                    else:
                        if not gpx_data['tzinfo']:
                            gpx_data['tzinfo'] = point.time.tzinfo

                        i_tstamp.append(i)

                    i += 1

    if i_tstamp and not len(i_latlon) == len(i_tstamp):
        for k in ('lat', 'lon', 'ele', 'tstamp'):
                gpx_data[k] = [gpx_data[k][i] for i in i_tstamp] if gpx_data[k] else None

    return gpx_data

def gpx_write(gpx_file, gpx_data, write_speed = False):
    # input: gpx_file = str
    #        gpx_data = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}
    #        write_speed = bool
    # output: None

    if write_speed:
        gpx_speed = gpx_calculate_speed(gpx_data)

    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx_segment = gpxpy.gpx.GPXTrackSegment()

    gpx.tracks.append(gpx_track)
    gpx_track.segments.append(gpx_segment)

    for i in range(len(gpx_data['lat'])):
        lat = gpx_data['lat'][i]
        lon = gpx_data['lon'][i]
        ele = gpx_data['ele'][i] if gpx_data['ele'] else None
        time = datetime.fromtimestamp(gpx_data['tstamp'][i], tz = gpx_data['tzinfo']) if gpx_data['tstamp'] else None
        speed = gpx_speed[i] if write_speed else None

        gpx_point = gpxpy.gpx.GPXTrackPoint(lat, lon, ele, time, speed = speed)

        gpx_segment.points.append(gpx_point)

    try:
        with open(gpx_file, 'w') as file:
            file.write(gpx.to_xml(version = '1.0' if write_speed else '1.1'))
    except:
        exit('ERROR Failed to save {}'.format(gpx_file))

    return

# main
def main():
    import argparse

    parser = argparse.ArgumentParser(description = 'interpolate GPX file(s) using linear or spline interpolation')

    parser.add_argument('gpx_files', metavar = 'FILE', nargs = '+', help = 'GPX file(s)')
    parser.add_argument('-d', '--deg', type = int, default = 1, help = 'interpolation degree, 1=linear, 2-5=spline (default: 1)')
    parser.add_argument('-r', '--res', type = float, default = 1, help = 'interpolation resolution in meters (default: 1)')
    parser.add_argument('-s', '--speed', action = 'store_true', help = 'Save interpolated speed')

    args = parser.parse_args()

    for gpx_file in args.gpx_files:
        if not '_interpolated.gpx' in gpx_file:
            gpx_data = gpx_read(gpx_file)

            print('Read {} trackpoints from {}'.format(len(gpx_data['lat']), gpx_file))

            gpx_data_interp = gpx_interpolate(gpx_data, args.res, args.deg)

            output_file = '{}_interpolated.gpx'.format(gpx_file[:-4])

            gpx_write(output_file, gpx_data_interp, write_speed = args.speed)

            print('Saved {} trackpoints to {}'.format(len(gpx_data_interp['lat']), output_file))

if __name__ == '__main__':
    main()