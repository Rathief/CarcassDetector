import os
import numpy as np
import pandas as pd
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


if __name__ == "__main__":
    track_vids = read_data("Simulated_Track")
    feature = []
    cls = []
    vid_len = len(track_vids)
    vid_count = 1
    for vid in track_vids:
        print(f"Processing Video ({vid_count}/{vid_len})")
        vid_count += 1
        vid_objs = extract_tracks(vid)
        for obj in vid_objs:
            avg = obj.obj_track["speed"].mean(axis="index")
            std = obj.obj_track["speed"].std(axis="index")
            # print(f'Speed every frame = {obj.obj_track["speed"]}\n'
            #       f'Average = {avg}, Standard Deviation = {std}')
            feature.append([avg, std])
            cls.append(obj.obj_class)
    df = pd.DataFrame(np.concatenate((feature, np.asarray([cls]).T), axis=1), columns=["mean", "std", "class"])
    output = open("ExtractionResult.csv", 'w')
    output.write(df.to_csv())
    output.close()
