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

def fill_neighboring_pixels(image,y,x,z,size):
    if x - 1 < size and x - 1 >= 0:
        image[y][x-1] = z
    if x + 1 < size and x + 1 >= 0:
        image[y][x+1] = z
    if y - 1 < size and y - 1 >= 0:
        image[y-1][x] = z
    if y + 1 < size and y + 1 >= 0:
        image[y+1][x] = z
    if y - 1 < size and y - 1 >= 0 and x - 1 < size and x - 1 >= 0:
        image[y-1][x-1] = z
    if y + 1 < size and y + 1 >= 0 and x + 1 < size and x + 1 >= 0:
        image[y+1][x+1] = z
    if y - 1 < size and y - 1 >= 0 and x + 1 < size and x + 1 >= 0:
        image[y-1][x+1] = z
    if y + 1 < size and y + 1 >= 0 and x - 1 < size and x - 1 >= 0:
        image[y+1][x-1] = z
    if x - 2 < size and x - 2 >= 0:
        image[y][x-2] = z
        if y - 1 < size and y - 1 >= 0:
            image[y-1][x-2] = z
        if y + 1 < size and y + 1 >= 0:
            image[y+1][x-2] = z
    if x + 2 < size and x + 2 >= 0:
        image[y][x+2] = z
        if y - 1 < size and y - 1 >= 0:
            image[y-1][x+2] = z
        if y + 1 < size and y + 1 >= 0:
            image[y+1][x+2] = z
    if y - 2 < size and y - 2 >= 0:
        image[y-2][x] = z
        if x - 1 < size and x - 1 >= 0:
            image[y-2][x-1] = z
        if x + 1 < size and x + 1 >= 0:
            image[y-2][x+1] = z
    if y + 2 < size and y + 2 >= 0:
        image[y+2][x] = z
        if x - 1 < size and x - 1 >= 0:
            image[y+2][x-1] = z
        if x + 1 < size and x + 1 >= 0:
            image[y+2][x+1] = z
    if y - 2 < size and y - 2 >= 0 and x - 2 < size and x - 2 >= 0:
        image[y-2][x-2] = z
    if y + 2 < size and y + 2 >= 0 and x + 2 < size and x + 2 >= 0:
        image[y+2][x+2] = z
    if y - 2 < size and y - 2 >= 0 and x + 2 < size and x + 2 >= 0:
        image[y-2][x+2] = z
    if y + 2 < size and y + 2 >= 0 and x - 2 < size and x - 2 >= 0:
        image[y+2][x-2] = z
    return image

def draw_line(image,size,previous_x, previous_y, pixel_x_position, pixel_y_position):
    line = list(bresenham(previous_x, previous_y, pixel_x_position, pixel_y_position))[1:-1]
    for pixel in line:
        y = pixel[1]
        x = pixel[0]
        if np.array_equal(image[y][x],[255.0,255.0,255.0]):
            image[y][x] = [85.0,0.0,81.0]
        if y-1 >= 0 and y-1 < size and x-1 >= 0 and x-1 < size:
            if np.array_equal(image[y-1][x-1],[255.0,255.0,255.0]):
                image[y-1][x-1] = [85.0,0.0,81.0]
            if np.array_equal(image[y][x-1],[255.0,255.0,255.0]):
                image[y][x-1] = [85.0,0.0,81.0]
            if np.array_equal(image[y-1][x],[255.0,255.0,255.0]):
                image[y-1][x] = [85.0,0.0,81.0]
        if y+1 >= 0 and y+1 < size and x+1 >= 0 and x+1 < size:
            if np.array_equal(image[y+1][x+1],[255.0,255.0,255.0]):
                image[y+1][x+1] = [85.0,0.0,81.0]
            if np.array_equal(image[y][x+1],[255.0,255.0,255.0]):
                image[y][x+1] = [85.0,0.0,81.0]
            if np.array_equal(image[y+1][x],[255.0,255.0,255.0]):
                image[y+1][x] = [85.0,0.0,81.0]
    return image

# Returns a BGR value based on z
def get_color_value(z,max_z):
    z_increment = max_z/11
    if z == -1.0:
        return [255.0,255.0,255.0]
    elif z <= z_increment:
        return [85.0,0.0,81.0]
    elif z <= z_increment*2:
        return [159.0,0.0,151.0]
    elif z <= z_increment*3:
        return [187.0,6.0,96.0]
    elif z <= z_increment*4:
        return [255.0,0.0,0.0]
    elif z <= z_increment*5:
        return [238.0,169.0,3.0]
    elif z <= z_increment*6:
        return [186.0,171.0,0.0]
    elif z <= z_increment*7:
        return [0.0,255.0,0.0]
    elif z <= z_increment*8:
        return [17.0,208.0,143.0]
    elif z <= z_increment*9:
        return [0.0,255.0,255.0]
    elif z <= z_increment*10:
        return [0.0,165.0,255.0]
    else:
        return [0.0,0.0,255.0]

def get_traclet(trajectory,size,max_speed):
    # scale coordinates to handle positions that do not differ significantly from each other
    trajectory['longitude'] = trajectory['longitude'].apply(lambda x: x*1000000000)
    trajectory['latitude'] = trajectory['latitude'].apply(lambda y: y*1000000000)
    # find the minima and maxima of the coordinates
    min_lon = trajectory['longitude'].min()
    max_lon = trajectory['longitude'].max()
    min_lat = trajectory['latitude'].min()
    max_lat = trajectory['latitude'].max()
    # calculate x and y distances
    dist_x = max_lon - min_lon
    dist_y = max_lat - min_lat
    image = np.ones((size, size, 3))*255
    if dist_x != 0 and dist_y != 0:
        previous_x = -1
        previous_y = -1
        # initialize an image representation -> height, width, channels
        for index, p in trajectory.iterrows():
            speed = p['speed']
            # get color value based on speed
            color_value = get_color_value(speed,max_speed)
            # find the position's x and y pixels
            pixel_x_position = max(min(int(((p['longitude'] - min_lon)/dist_x)*size),size-1),0.0)
            pixel_y_position = max(min(size - int(((p['latitude'] - min_lat)/dist_y)*size),size-1),0.0)
            # color the pixel
            image[pixel_y_position][pixel_x_position] = color_value
            # color neighboring pixels to add thickness
            image = fill_neighboring_pixels(image,pixel_y_position,pixel_x_position,color_value,size)
            # draw a thick line between the previous and the current pixel
            if previous_x != -1 and previous_y != -1:
                image = draw_line(image,size,previous_x, previous_y, pixel_x_position, pixel_y_position)
            previous_x = pixel_x_position
            previous_y = pixel_y_position
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
makedirs('traclets', exist_ok=True)

print(f'[INFO] Converting {num_trajectories} trajectories to images...')
# Initial call to print 0% progress
printProgressBar(0, num_trajectories, prefix = 'Progress:', suffix = 'Complete')

times = []
idx = 0
for t in trajectories:
    id = t['trajectory_id'].iloc[0]
    label = t['label'].iloc[0]
    # make folder for labels
    makedirs(f'traclets/{label}', exist_ok=True)
    start = datetime.datetime.now()
    traclet = get_traclet(t,size,max_speed)
    end = datetime.datetime.now()
    cv2.imwrite(f'traclets/{label}/{id}.png', traclet)
    delta = end - start
    ms = int(delta.total_seconds() * 1000)
    times.append(ms)
    printProgressBar(idx + 1, num_trajectories, prefix = 'Progress:', suffix = 'Complete')
    idx += 1

print(f'[INFO] Done.')
print(f'[INFO] Average traclet creation time: {sum(times)/len(times)} ms.')