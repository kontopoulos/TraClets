from os import makedirs
import argparse
import datetime
import numpy as np
import pandas as pd
from bresenham import bresenham
import cv2

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Returns a BGR value based on z
def get_color_value(speed,lon,lat):
    return [float(speed),float(lon),float(lat)]

def get_triangles(lat,lon):
    pointsmap = []
    lat = int(round(lat))
    lon = int(round(lon))
    for point in list(bresenham(lat, lon, lon, lat)):
        pointsmap.append(point)
    for point in list(bresenham(lat, lon, 2*lon-1,2*lat-1)):
        pointsmap.append(point)
    for point in list(bresenham(lon, lat, 2*lon-1,2*lat-1)):
        pointsmap.append(point)
        
    return pointsmap
        
def scale(min,max,current,target):
    return int(round((target*(current-min)) / (max - min)))

def get_traclet(trajectory,size,max_speed):
    # find the minima and maxima of the coordinates
    min_lon = trajectory['longitude'].min()
    max_lon = trajectory['longitude'].max()
    min_lat = trajectory['latitude'].min()
    max_lat = trajectory['latitude'].max()
    image = np.ones((size, size, 3))*255
    if min_lon != max_lon and min_lat != max_lat:
        # initialize an image representation -> height, width, channels
        for index, p in trajectory.iterrows():
            speed = p['speed']
            # get color value based on speed
            color_value = get_color_value(scale(0,max_speed,speed,255),scale(min_lon,max_lon,p['longitude'],255),scale(min_lat,max_lat,p['latitude'],255))
            # find the position's x and y pixels
            for y_x in get_triangles(scale(min_lon,max_lon,p['longitude'],int(round((size-1)/2))),scale(min_lat,max_lat,p['latitude'],int(round((size-1)/2)))):
                image[y_x[0]][y_x[1]] = [int(round((col + image[y_x[0]][y_x[1]][col_index]) / 2 )) for col_index,col in enumerate(color_value)]
        return np.array(image).astype('uint8')
    else:
        return np.array(image).astype('uint8')


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-s", "--size", type=int, help="path and name to output model")
args = vars(ap.parse_args())

dataset = pd.read_csv(args["dataset"])
size = args["size"]
num_trajectories = len(set(dataset['trajectory_id']))
max_speed = dataset['speed'].max()
trajectories = [g[1] for g in list(dataset.groupby("trajectory_id"))]

# make folder for results
makedirs('traclets_triangle_image', exist_ok=True)

print(f'[INFO] Converting {num_trajectories} trajectories to images...')
# Initial call to print 0% progress
printProgressBar(0, num_trajectories, prefix = 'Progress:', suffix = 'Complete')

times = []
idx = 0
for t in trajectories:
    id = t['trajectory_id'].iloc[0]
    label = t['label'].iloc[0]
    # make folder for labels
    makedirs(f'traclets_triangle_image/{label}', exist_ok=True)
    start = datetime.datetime.now()
    traclet = get_traclet(t,size,max_speed)
    end = datetime.datetime.now()
    cv2.imwrite(f'traclets_triangle_image/{label}/{id}.png', traclet)
    delta = end - start
    ms = int(delta.total_seconds() * 1000)
    times.append(ms)
    printProgressBar(idx + 1, num_trajectories, prefix = 'Progress:', suffix = 'Complete')
    idx += 1

print(f'[INFO] Done.')
print(f'[INFO] Average traclet creation time: {sum(times)/len(times)} ms.')