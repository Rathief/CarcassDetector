import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class PyplotWrapper:
    def __init__(self, xlabel='', ylabel='', title='', figsize=(8, 6)):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

    def plot(self, x, y, label=None, **kwargs):
        self.ax.plot(x, y, label=label, **kwargs)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.ax.legend()

    def scatter(self, x, y, label=None, **kwargs):
        self.ax.scatter(x, y, label=label, **kwargs)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.ax.legend()

    def show(self):
        plt.show()
def read_data(data_folder):
    data = []
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        if os.path.isfile(filepath):
            track_text = open(filepath, 'r')
            track_text_array = []
            for line in track_text:
                track_text_array.append(map(int, line.split()))
            track_text.close()
            data.append(track_text_array)
    return data


def extract_tracks(track_array):
    tracks_by_frames = pd.DataFrame(data=track_array,
                                    columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "class"]
                                    )
    # turns into [id, class, track]
    # where track = [frame, [bb]]
    tracks_by_id = []
    for object_id in tracks_by_frames['id'].unique():
        object_class = tracks_by_frames[tracks_by_frames['id'] == object_id]['class'].iloc[0]
        object_track = tracks_by_frames[tracks_by_frames['id'] == object_id].reindex(columns=
                                                                                     ["frame", "bb_left", "bb_top",
                                                                                      "bb_width", "bb_height"])
        object_data = ObjectTracking(object_id, object_class, object_track)
        tracks_by_id.append(object_data)
    return tracks_by_id  # list of objects


# balance the data so dead and living chickens are equally frequent
def balance_data(data, class_col):
    pass


class ObjectTracking:
    def __init__(self, obj_id, obj_class, obj_track):
        x_column = []
        y_column = []
        for frame in range(obj_track.shape[0]):
            obj_x = obj_track["bb_left"].iloc[frame] + (obj_track["bb_width"].iloc[frame] / 2)
            obj_y = obj_track["bb_top"].iloc[frame] + (obj_track["bb_height"].iloc[frame] / 2)
            x_column.append(obj_x)
            y_column.append(obj_y)
        vx = [0]
        vy = [0]
        speed = [0]
        for i in range(len(x_column)-1):
            x_speed = (x_column[i+1] - x_column[i]) / (obj_track["frame"].iloc[i+1] - obj_track["frame"].iloc[i])
            y_speed = (y_column[i+1] - y_column[i]) / (obj_track["frame"].iloc[i+1] - obj_track["frame"].iloc[i])
            vx.append(x_speed)
            vy.append(y_speed)
            speed.append(np.sqrt(pow(x_speed, 2) + pow(y_speed, 2)))

        obj_track["bb_x"] = x_column
        obj_track["bb_y"] = y_column
        obj_track["vx"] = vx
        obj_track["vy"] = vy
        obj_track["speed"] = speed
        self.obj_id = obj_id
        self.obj_class = obj_class
        self.obj_track = obj_track

    def print_properties(self):
        print("Object ID =" + str(self.obj_id))
        print("Object Class =" + str(self.obj_class))
        print(self.obj_track)


# neighbor_classifiers = [
#     KNeighborsClassifier(n_neighbors=5, weights="uniform"),
#     KNeighborsClassifier(n_neighbors=10, weights="uniform"),
#     KNeighborsClassifier(n_neighbors=10, weights="distance")
# ]
#
# tree_classifiers = [
#     RandomForestClassifier(max_depth=None, class_weight=None),  # default
#     RandomForestClassifier(max_depth=3, class_weight=None),
#     RandomForestClassifier(max_depth=None, class_weight="balanced")
# ]
#
# svc_classifiers = [
#     SVC(kernel="rbf", class_weight=None),
#     SVC(kernel="poly", class_weight=None),
#     SVC(kernel="rbf", class_weight="balanced")
# ]
clf_params = [[KNeighborsClassifier(), {'n_neighbors': [3, 5, 10, 20], 'weights': ['uniform', 'distance']}],
              [RandomForestClassifier(), {'max_depth': [None, 3, 5], 'class_weight': [None, 'balanced']}],
              [SVC(), {'kernel': ['rbf', 'poly'], 'class_weight': [None, 'balanced']}]
              ]


if __name__ == '__main__':
    track_vids = read_data("Simulated_Track")
    X = []
    y = []
    for vid in track_vids:
        for obj in extract_tracks(vid):
            #  skips object with only one frame of appearance
            if obj.obj_track.shape[0] <= 1:
                continue
            avg = obj.obj_track["speed"].mean(axis="index")
            std = obj.obj_track["speed"].std(axis="index")
            X.append([avg, std])
            y.append(obj.obj_class)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    best_classifiers = []
    plotnum = 1
    for clf in clf_params:
        pltneg = []
        pltpos = []
        grid_search = GridSearchCV(estimator=clf[0], param_grid=clf[1], scoring='balanced_accuracy')
        grid_search.fit(X_train, y_train)
        # pd.DataFrame(grid_search.cv_results_).to_excel(clf[0].__class__.__name__ + '.xlsx')
        y_pred = grid_search.best_estimator_.predict(X_test)
        # for i in range(len(y_test)):
        #     print(f"{y_test[i]} {y_pred[i]}")
        pltneg.append([X_test[i][0] for i in range(len(X_test)) if y_pred[i] == 0])
        pltneg.append([X_test[i][1] for i in range(len(X_test)) if y_pred[i] == 0])
        pltpos.append([X_test[i][0] for i in range(len(X_test)) if y_pred[i] == 1])
        pltpos.append([X_test[i][1] for i in range(len(X_test)) if y_pred[i] == 1])
        plt.subplot(4, 4, plotnum)
        plt.hist(pltneg[0])
        plotnum += 1
        plt.subplot(4, 4, plotnum)
        plt.hist(pltneg[1])
        plotnum += 1
        plt.subplot(4, 4, plotnum)
        plt.hist(pltpos[0])
        plotnum += 1
        plt.subplot(4, 4, plotnum)
        plt.hist(pltpos[1])
        plotnum += 1
        print(f"Classification report for {clf[0].__class__.__name__} : \n{classification_report(y_test, y_pred)}")
    realneg = [[X_test[i][0] for i in range(len(X_test)) if y_test[i] == 0],
               [X_test[i][1] for i in range(len(X_test)) if y_test[i] == 0]
               ]
    realpos = [[X_test[i][0] for i in range(len(X_test)) if y_test[i] == 1],
               [X_test[i][1] for i in range(len(X_test)) if y_test[i] == 1]
               ]
    plt.subplot(4, 4, plotnum)
    plt.hist(realneg[0])
    plotnum += 1
    plt.subplot(4, 4, plotnum)
    plt.hist(realneg[1])
    plotnum += 1
    plt.subplot(4, 4, plotnum)
    plt.hist(realpos[0])
    plotnum += 1
    plt.subplot(4, 4, plotnum)
    plt.hist(realpos[1])
    plt.show()

