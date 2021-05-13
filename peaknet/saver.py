import numpy as np

class Saver():
    def __init__(self, saver_type, params):
        self.saver_type = saver_type
        if saver_type=="precision_recall":
            print("Precision and recall will be saved in saved_outputs directory")
            self.content = {"params": params, "precision": [], "recall": []}
        elif saver_type is None:
            print("Data will not be save in saved_outputs directory.")

    def upload(self, data):
        if self.saver_type == "precision_recall":
            self.content["precision"].append(float(data["precision"].data.cpu()))
            self.content["recall"].append(float(data["recall"].data.cpu()))

    def save(self, save_name):
        if self.saver_type is not None:
            filename = 'saved_outputs/' + save_name + '.npy'
            print("Saving at " + filename)
            np.save(filename, self.content)
            print("Saved!")