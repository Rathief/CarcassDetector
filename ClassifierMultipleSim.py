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

STEP = 10       # 10% steps
LIMIT1 = 100     # Maximum percentage
LIMIT2 = 50

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
    filelist = os.listdir(data_folder)
    fl_len = len(filelist)
    flcount = 1
    for filename in filelist:
        print(f"Reading File ({flcount}/{fl_len})")
        flcount += 1
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
    unique_objs = tracks_by_frames['id'].unique()
    unique_len = len(unique_objs)
    unique_count = 1
    for object_id in unique_objs:
        print(f"     Extracting Object ({unique_count}/{unique_len})")
        unique_count += 1
        unique_object = tracks_by_frames[tracks_by_frames['id'] == object_id]
        # Skips object with only a single frame of occurence
        if unique_object.shape[0] <= 1:
            continue
        object_class = unique_object['class'].iloc[0]
        object_track = unique_object.reindex(columns=
                                                                                     ["frame", "bb_left", "bb_top",
                                                                                      "bb_width", "bb_height"])
        object_data = ObjectTracking(object_id, object_class, object_track)
        tracks_by_id.append(object_data)
    return tracks_by_id  # list of objects


# generate data with similar distribution
def generate_similar(mean, var, n):
    gen = []
    for i in range(n):
        gen.append(
            np.abs(
                np.random.default_rng(i).normal(loc=mean, scale=var)
            ).tolist()
        )
    return gen


def calc_mean_std(data):
    array_data = np.asarray(data)
    data_mean = np.mean(array_data, axis=0)
    data_std = np.std(array_data, axis=0)
    return data_mean, data_std


class ObjectTracking:
    def __init__(self, obj_id, obj_class, obj_track):
        x_column = []
        y_column = []
        for frame in range(obj_track.shape[0]):
            obj_x = obj_track["bb_left"].iloc[frame] + (obj_track["bb_width"].iloc[frame] / 2)
            obj_y = obj_track["bb_top"].iloc[frame] + (obj_track["bb_height"].iloc[frame] / 2)
            x_column.append(obj_x)
            y_column.append(obj_y)
        vx_array = [0]
        vy_array = [0]
        speed = [0]
        for i in range(len(x_column)-1):
            vx = (x_column[i+1] - x_column[i]) / (obj_track["frame"].iloc[i+1] - obj_track["frame"].iloc[i])
            vy = (y_column[i+1] - y_column[i]) / (obj_track["frame"].iloc[i+1] - obj_track["frame"].iloc[i])
            vx_array.append(vx)
            vy_array.append(vy)
            speed.append(np.sqrt(pow(vx, 2) + pow(vy, 2)))

        obj_track["bb_x"] = x_column
        obj_track["bb_y"] = y_column
        obj_track["vx"] = vx_array
        obj_track["vy"] = vy_array
        obj_track["speed"] = speed
        self.obj_id = obj_id
        self.obj_class = obj_class
        self.obj_track = obj_track

    def print_properties(self):
        print("Object ID =" + str(self.obj_id))
        print("Object Class =" + str(self.obj_class))
        print(self.obj_track)

clf_params = [[KNeighborsClassifier(),      {'n_neighbors': [3, 5, 10, 20],
                                             'weights': ['uniform', 'distance']}],
              [RandomForestClassifier(),    {'n_estimators': [2, 5, 10],
                                             'max_depth': [None, 3, 5],
                                             'class_weight': [None, 'balanced']}],
               [SVC(),                       {'C': [0.1, 1, 10, 100],
                                             'kernel': ['rbf', 'sigmoid'],
                                             'class_weight': [None, 'balanced']}]
              ]


if __name__ == '__main__':
    extractionResult = pd.read_csv("ExtractionResult.csv")

    count = 0
    living = extractionResult.loc[extractionResult["class"] == 0, ["mean", "std"]]  # DF of living chicken's feature
    dead = extractionResult.loc[extractionResult["class"] == 1, ["mean", "std"]]    # DF of dead chicken's feature
    feature = extractionResult.loc[:, ["mean", "std"]].values.tolist()  # List of features
    cls = extractionResult.loc[:, "class"].values.tolist()              # List of class
    livingMean, livingStd = calc_mean_std(living.values.tolist())  # Calculate mean and std of the features
    deadMean, deadStd = calc_mean_std(dead.values.tolist())
    # Run fitting multiple times until std dead = std living OR best accuracy <= 80%
    while count <= LIMIT1:
        cur_mean = deadMean - (count/100)*(deadMean-livingMean)
        cur_std = deadStd - (count/100)*(deadStd-livingStd)
        print(f"Testing at {count}% similarity, current std = {cur_std}")
        print("Generating Simulated data")
        rand_normal = generate_similar(cur_mean, cur_std, round(extractionResult.shape[0]/2))
        X = feature + rand_normal
        y = cls + [1]*(len(rand_normal))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
        classifier_accs = []
        for clf in clf_params:
            print(f"Current Classifier: {clf[0].__class__.__name__}")
            grid_search = GridSearchCV(estimator=clf[0], param_grid=clf[1], scoring='balanced_accuracy', verbose=2)
            print(f"Fitting Classifier...")
            grid_search.fit(X_train, y_train)
            print(f"Classifier: {clf[0].__class__.__name__}\n"
                  f"Best Parameters: {grid_search.best_params_}\n"
                  f"Best Score: {grid_search.best_score_}\n"
                  f"Refitting Time: {grid_search.refit_time_}")
            # pd.DataFrame(grid_search.cv_results_).to_excel(clf[0].__class__.__name__ + '.xlsx')
            print(f"Testing Classifier...")
            y_pred = grid_search.best_estimator_.predict(X_test)
            print(f"Classification report for {clf[0].__class__.__name__} : \n{classification_report(y_test, y_pred)}")
            classifier_accs.append(classification_report(y_test, y_pred, output_dict=True)['accuracy'])
        if max(classifier_accs) <= 0.8:
            break
        count += STEP
    plotter = PyplotWrapper(xlabel='mean', ylabel='std', figsize=(1,1))
    if count > LIMIT1:
        print(f"Accuracy is still higher than 80% in spite of {LIMIT1}% similarity of deviation")
    else:
        print(f"Greatest Accuracy of {max(classifier_accs)} is reached at deviation similarity of {count}%")
    plotter.scatter([living["mean"].values.tolist()], living["std"].values.tolist(), marker=".", label="Living")
    plotter.scatter(dead["mean"].values.tolist(), dead["std"].values.tolist(), marker=".", label="Perfect Dead")
    plotter.scatter([sublist[0] for sublist in rand_normal], [sublist[1] for sublist in rand_normal], marker=".", label="Imperfect Dead")
    plotter.show()

