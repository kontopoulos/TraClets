## TraClets


<p align="center">
  <img width="846" height="391" src="https://github.com/kontopoulos/TraClets/blob/main/traclet.png" alt="Sublime's custom image"/>
</p>


### What is a TraClet?

A traclet is an image representation of a trajectory. This representation is indicative of the mobility patterns of the moving objects. TraClets need to efficiently visualize and capture two key features that characterize the trajectory patterns of moving objects: i) the shape of the trajectory which indicates the way the object moves in space, and ii) the speed that indicates how fast the object moves in space.

---

#### Example Usage


###### _General cmd_
```shell
python traclet.py --d [dataset_path] --s [size of the resulting images]
```

###### _A local example cmd_
```shell
python traclet.py --d dataset.csv --s 224
```

After the execution, a folder called traclets is created that contains one folder per label in the dataset. Each label folder contains the respective trajectory images.

##### Dependencies

- numpy
- pandas
- opencv-python
- bresenham

---

#### Classification Example

The repository also contains a file that uses the RandomForests classifier to perform k-fold cross-validation on the generated traclets. The features used for the classification are:

- Color histogram
- Hu invariant moments

###### _General cmd_
```shell
python classifer.py --d [dataset_path] --f [number of folds]
```

###### _A local example cmd_
```shell
python classifer.py --d traclets --f 5
```

##### Dependencies

- numpy
- opencv-python
- imutils
- scikit-learn

---

#### Relevant Repositories

- [d-LOOK](https://github.com/AntonisMakris/d-LOOK)

---

### References

\[[1]\](https://ieeexplore.ieee.org/abstract/document/9474859). I. Kontopoulos, A. Makris, D. Zissis, K. Tserpes, "A computer vision approach for trajectory classification", 22nd IEEE International Conference on Mobile Data Management (MDM), 2021.  
\[[2]\](https://www.mdpi.com/2220-9964/10/4/250). I. Kontopoulos, A. Makris, K. Tserpes, "A Deep Learning Streaming Methodology for Trajectory Classification", ISPRS International Journal of Geo-Information, Volume 10, Issue 4, 2021.