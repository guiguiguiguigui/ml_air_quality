import numpy as np
import math
import random
import data_merge as dr

FEATURES = ["d" + str(i) for i in range(16)]
OUTPUT = ["no"]

def process(input_indices, output_indices, data_point):
    input = []
    output = []

    for i in sorted(list(input_indices)):
        input.append(float(data_point[i]))
    for i in sorted(list(output_indices)):
        if float(data_point[i]) < 0:
            raise ValueError
        output.append(float(data_point[i]))

    return (input, output)


def get_data():
    d = dr.read_data("alliance_sub.csv")

    output_indices = set()
    input_indices = set()

    
    for j in range(len(d[0])):
        for i in range(len(FEATURES)):
            if FEATURES[i] in d[0][j]:
                input_indices.add(j)
        for i in range(len(OUTPUT)):
            if OUTPUT[i] in d[0][j]:
                if "no2" in d[0][j]:
                    continue
                output_indices.add(j)

    print (input_indices)
    print (output_indices)
   
    points = []

    for point in d[1:]:
        try:
            points.append(process(input_indices, output_indices, point))
        except:
            continue

    random.shuffle(points)

    print (len(points))

    train_data = points[:4000]
    validation_data = points[4000:5000]
    test_data = points[5000:6000]

    train_X = []
    train_Y = []

    test_X = []
    test_Y = []

    dev_X = []
    dev_Y = []

    for point in train_data:
        train_X.append(point[0])
        train_Y.append(point[1])

    for point in test_data:
        test_X.append(point[0])
        test_Y.append(point[1])

    for point in validation_data:
        dev_X.append(point[0])
        dev_Y.append(point[1])

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    dev_X = np.array(dev_X)
    dev_Y = np.array(dev_Y)

    mapping = {"train":(train_X, train_Y), "test":(test_X, test_Y), \
            "dev":(dev_X, dev_Y)}
    return mapping

if __name__ == "__main__":
    data = get_data()
    print (data["train"])
